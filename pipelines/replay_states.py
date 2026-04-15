"""
Replay saved round logs (prompt_dict + output_dict pickles) to rebuild
_AgentState objects and save a states pickle, WITHOUT making any API calls
(no Brave, no ZenRows, no generate, no DB lookups).

Usage:
    python replay_states.py <round_log_dir> <prompt_dict_pickle>
        [--cutoff_date DATE] [--lower_bound N] [--upper_bound N]

Example:
    python replay_states.py \
        /path/to/round_logs \
        /path/to/agent_prompt.pickle \
        --cutoff_date 2024-09-01 --lower_bound 50000 --upper_bound 150000
"""

import os
import sys
import re
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from run_trial_agent_multithread import (
    _AgentState, _get_encoding, _build_token_reminder,
    parse_output, _extract_summary, _extract_strategy, _extract_explain,
    _build_history_entry, _check_missing_keywords,
    FORMAT_ERROR_TEMPLATE, NON_COMPLIANCE_ERROR_TEMPLATE, PARSE_FUNCTION_CODE,
    count_tokens, PROTOCOL_TEMPLATE,
)


TOKEN_REMINDER_RE = re.compile(
    r"Environment Input:\n\n"
    r"Token reminder: prior_tokens=\d+, current_input_tokens=\d+, total_tokens=\d+\.\n\n"
    r"The lower_bound is \d+ token, the upper bound is \d+ token\.\n\n"
)


def _find_max_round(round_log_dir):
    """Find the highest round number that has both prompt and output pickles."""
    max_round = 0
    for fname in os.listdir(round_log_dir):
        m = re.match(r'round_(\d+)_output_dict\.pickle$', fname)
        if m:
            r = int(m.group(1))
            prompt_path = os.path.join(round_log_dir, f"round_{r}_prompt_dict.pickle")
            if os.path.exists(prompt_path):
                max_round = max(max_round, r)
    return max_round


def _extract_current_input_parts(next_prompt, conversation_history, filled_protocol, trial_question):
    """
    Given the next round's full prompt, strip off the known prefix
    (conversation_history + protocol + question) and the token reminder
    to recover current_input_prefix + current_input_body.
    """
    # For round 1, the full prompt IS current_input (prefix + reminder + body)
    # For round N>1, full_prompt = conv_history + protocol + "\n\n" + question + "\n\n" + prefix + reminder + body
    known_prefix = conversation_history + filled_protocol + "\n\n" + trial_question + "\n\n"

    if next_prompt.startswith(known_prefix):
        remainder = next_prompt[len(known_prefix):]
    else:
        # Round 1 case or mismatch — treat full prompt as the current input
        remainder = next_prompt

    # Strip token reminder from remainder
    m = TOKEN_REMINDER_RE.search(remainder)
    if m:
        # Everything before the token reminder is current_input_prefix
        prefix = remainder[:m.start()]
        body = remainder[m.end():]
        return prefix, body
    else:
        # No token reminder found — treat entire remainder as body
        return "", remainder


def replay(round_log_dir, key2question, agent_kwargs):
    """
    Replay saved round logs to rebuild states without any API calls.
    Returns the states dict and the last replayed round number.
    """
    enc = _get_encoding()

    states = {
        key: _AgentState(key, question, **agent_kwargs)
        for key, question in key2question.items()
    }

    max_round = _find_max_round(round_log_dir)
    if max_round == 0:
        print("No complete round logs found.")
        return states, 0

    print(f"Found round logs up to round {max_round}")

    # Pre-load all prompt and output dicts
    all_prompts = {}
    all_outputs = {}
    for r in range(1, max_round + 1):
        prompt_path = os.path.join(round_log_dir, f"round_{r}_prompt_dict.pickle")
        output_path = os.path.join(round_log_dir, f"round_{r}_output_dict.pickle")
        with open(prompt_path, 'rb') as f:
            all_prompts[r] = pickle.load(f)
        with open(output_path, 'rb') as f:
            all_outputs[r] = pickle.load(f)
        # Also load retry outputs if available
        retry_output_path = os.path.join(round_log_dir, f"round_{r}_retry_output_dict.pickle")
        if os.path.exists(retry_output_path):
            with open(retry_output_path, 'rb') as f:
                retry_dict = pickle.load(f)
                # Merge retry into main output (retry overrides None)
                for k, v in retry_dict.items():
                    if v is not None and all_outputs[r].get(k) is None:
                        all_outputs[r][k] = v

    for r in range(1, max_round + 1):
        print(f"  Replaying round {r}...")
        prompt_dict = all_prompts[r]
        output_dict = all_outputs[r]

        # Next round's prompt dict (if available) for extracting current_input_body
        next_prompt_dict = all_prompts.get(r + 1)

        active_keys = [k for k, s in states.items() if not s.finished and k in prompt_dict]

        for key in active_keys:
            state = states[key]
            output = output_dict.get(key)

            # Advance round_num
            state.round_num += 1

            # Compute current_input_tokens from the saved prompt
            saved_prompt = prompt_dict[key]
            if state.round_num == 1:
                current_input_tokens = count_tokens(saved_prompt, enc)
            else:
                # The "current input" portion = full_prompt - (conv_history + protocol + "\n\n" + question + "\n\n")
                known_prefix = (
                    state.conversation_history
                    + state.filled_protocol + "\n\n"
                    + state.trial_question + "\n\n"
                )
                if saved_prompt.startswith(known_prefix):
                    current_input_str = saved_prompt[len(known_prefix):]
                else:
                    current_input_str = saved_prompt
                current_input_tokens = count_tokens(current_input_str, enc)

            state._current_input_tokens = current_input_tokens
            state._total_tokens = state.prior_tokens + current_input_tokens

            if output is None:
                # Model returned None
                state.none_count += 1
                if state.none_count > 3:
                    state.finished = True
                    state.result = {'error': 'Model returned None', 'key': key}
                state.prior_tokens += current_input_tokens
                continue

            # Parse output
            try:
                action, value, extra = parse_output(output)
            except ValueError as e:
                # Format error — update history and set error input for next round
                missing = _check_missing_keywords(output)
                missing_str = ", ".join(missing) if missing else "None"
                error_msg = (
                    FORMAT_ERROR_TEMPLATE
                    .replace("{curr_output}", output)
                    .replace("{parsing_function_code}", PARSE_FUNCTION_CODE)
                    .replace("{error}", str(e))
                    .replace("{missing_keywords}", missing_str)
                )
                state.conversation_history += _build_history_entry(state.round_num, output)
                state.prior_tokens += current_input_tokens
                state.current_input_body = error_msg
                state.current_input_prefix = ""
                state.prev_action = None
                continue

            # Check missing required keywords
            missing_keywords = _check_missing_keywords(output, action=action)
            if missing_keywords:
                missing_str = ", ".join(missing_keywords)
                non_compliance_kw = (
                    f"Missing required keywords: {missing_str}. "
                    f"Rule 5.1 requires STRATEGY: at every turn. "
                    f"Rule 5.2 requires SUMMARY: at every turn. "
                    f"Rule 11 requires EXPLAIN: at every turn. "
                    f"Rule 6 requires ANSWER: and REASON: when answering. "
                    f"Rule 7 requires QUERY: when submitting a query. "
                    f"Rule 8 requires URL: when requesting a webpage."
                )
                error_msg = (
                    NON_COMPLIANCE_ERROR_TEMPLATE
                    .replace("{curr_output}", output)
                    .replace("{assertion_error}", non_compliance_kw)
                )
                state.conversation_history += _build_history_entry(state.round_num, output, action, value)
                state.prior_tokens += current_input_tokens
                state.current_input_body = error_msg
                state.current_input_prefix = ""
                state.prev_action = None
                continue

            # Non-compliance checks
            total_tokens = state._total_tokens
            non_compliance = None
            if action == 'ANSWER':
                if total_tokens < state.lower_bound:
                    non_compliance = (
                        f"Rule 1 violated: You answered with total_tokens={total_tokens} "
                        f"which is less than lower_bound={state.lower_bound}. "
                        f"You CANNOT answer before processing {state.lower_bound} tokens."
                    )
            elif action == 'QUERY':
                if total_tokens > state.upper_bound:
                    non_compliance = (
                        f"Rule 2 violated: total_tokens={total_tokens} exceeds "
                        f"upper_bound={state.upper_bound}. You MUST answer the question now."
                    )
            elif action == 'URL':
                if total_tokens > state.upper_bound:
                    non_compliance = (
                        f"Rule 2 violated: total_tokens={total_tokens} exceeds "
                        f"upper_bound={state.upper_bound}. You MUST answer the question now."
                    )

            if non_compliance:
                error_msg = (
                    NON_COMPLIANCE_ERROR_TEMPLATE
                    .replace("{curr_output}", output)
                    .replace("{assertion_error}", non_compliance)
                )
                state.conversation_history += _build_history_entry(state.round_num, output, action, value)
                state.prior_tokens += current_input_tokens
                state.current_input_body = error_msg
                state.current_input_prefix = ""
                state.prev_action = None
                continue

            # Valid action
            if action == 'ANSWER':
                assert isinstance(value, tuple) and all(isinstance(p, float) for p in value)
                state.finished = True
                state.result = {
                    'answer': value,
                    'reason': extra,
                    'rounds': state.round_num,
                    'total_tokens': total_tokens,
                    'key': key,
                }
                state.prior_tokens += current_input_tokens
                continue

            # Update conversation history (same as process_output_phase1)
            summary = _extract_summary(output)
            strategy = _extract_strategy(output)
            explain = _extract_explain(output)
            if action == 'QUERY':
                action_str = f"query:{value}"
            elif action == 'URL':
                action_str = f"url:{value}"
            state.conversation_history += (
                f"My output for round {state.round_num}:\n"
                f"{action_str}\n"
                f"My strategy for round {state.round_num}:\n"
                f"{strategy}\n"
                f"My intention and explanation for my output for round {state.round_num}:\n"
                f"{explain}\n"
                f"My summary for round {state.round_num}'s environment input, "
                f"which is the environment's response to my output above:\n"
                f"{summary}\n"
            )
            state.prior_tokens += current_input_tokens

            # Set current_input_body/prefix from next round's prompt (no API calls)
            if action == 'QUERY':
                state.total_brave_queries += 1
                state.current_query = value
                state.prev_action = 'QUERY'
            elif action == 'URL':
                state.visited_urls.add(value)
                state.prev_action = 'URL'

            if next_prompt_dict and key in next_prompt_dict:
                # Extract current_input_body from the next round's saved prompt
                prefix, body = _extract_current_input_parts(
                    next_prompt_dict[key],
                    state.conversation_history,
                    state.filled_protocol,
                    state.trial_question,
                )
                state.current_input_prefix = prefix
                state.current_input_body = body
            else:
                # Last round — no next prompt available.
                # Leave current_input_body/prefix as-is; the main program
                # will call build_prompt which will redo the action (brave/URL).
                # Set a marker so the main program knows to redo the action.
                if action == 'QUERY':
                    state._pending_query = value
                    state.current_input_body = f"[REPLAY: brave search pending for query: {value}]"
                    state.current_input_prefix = ""
                elif action == 'URL':
                    state.current_input_body = f"[REPLAY: URL fetch pending for: {value}]"
                    state.current_input_prefix = ""
                print(f"    WARNING: Round {r} is the last round for key={key}, "
                      f"action={action}. State may need brave/URL fetch at resume.")

    # Save states pickle for the last round
    states_save_path = os.path.join(round_log_dir, f"round_{max_round}_states.pickle")
    states_to_save = {}
    for key, state in states.items():
        state_dict = {attr: getattr(state, attr) for attr in _AgentState.__slots__ if attr != 'enc' and hasattr(state, attr)}
        states_to_save[key] = state_dict
    with open(states_save_path, 'wb') as f:
        pickle.dump(states_to_save, f)
    print(f"\nSaved states pickle: {states_save_path}")
    print(f"  Keys: {len(states_to_save)}")
    print(f"  Finished: {sum(1 for s in states.values() if s.finished)}")
    print(f"  Active: {sum(1 for s in states.values() if not s.finished)}")

    return states, max_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay round logs to rebuild states pickle (no API calls)")
    parser.add_argument("round_log_dir", type=str, help="Directory containing round_N_*.pickle files")
    parser.add_argument("prompt_dict_pickle", type=str, help="Path to the original prompt_dict pickle (key2question)")
    parser.add_argument("--cutoff_date", type=str, default="2024-09-01")
    parser.add_argument("--lower_bound", type=int, default=50000)
    parser.add_argument("--upper_bound", type=int, default=60000)
    args = parser.parse_args()

    with open(args.prompt_dict_pickle, 'rb') as f:
        key2question = pickle.load(f)
    print(f"Loaded {len(key2question)} questions from {args.prompt_dict_pickle}")

    agent_kwargs = {
        'cutoff_date': args.cutoff_date,
        'lower_bound': args.lower_bound,
        'upper_bound': args.upper_bound,
    }

    replay(args.round_log_dir, key2question, agent_kwargs)
