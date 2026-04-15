# exp_name="brave_exp1"
# query_type="brief_title"
# input_file="./mar13_pipeline_data.pickle"
# freshness_flag="True"
# cutoff_date="2026-03-01"

# echo "exp_name: ${exp_name}"
# echo "query_type: ${query_type}"
# echo "input_file: ${input_file}"
# echo "freshness_flag: ${freshness_flag}"
# echo "cutoff_date: ${cutoff_date}"
# python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

# exp_name="brave_exp2"
# query_type="brief_title"
# input_file="./mar13_pipeline_data.pickle"
# freshness_flag="False"
# cutoff_date="2026-03-01"

# echo "exp_name: ${exp_name}"
# echo "query_type: ${query_type}"
# echo "input_file: ${input_file}"
# echo "freshness_flag: ${freshness_flag}"
# echo "cutoff_date: ${cutoff_date}"
# python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

# exp_name="brave_exp3"
# query_type="nct_id"
# input_file="./mar13_pipeline_data.pickle"
# freshness_flag="True"
# cutoff_date="2026-03-01"

# echo "exp_name: ${exp_name}"
# echo "query_type: ${query_type}"
# echo "input_file: ${input_file}"
# echo "freshness_flag: ${freshness_flag}"
# echo "cutoff_date: ${cutoff_date}"
# python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

# exp_name="brave_exp4"
# query_type="nct_id"
# input_file="./mar13_pipeline_data.pickle"
# freshness_flag="False"
# cutoff_date="2026-03-01"

# echo "exp_name: ${exp_name}"
# echo "query_type: ${query_type}"
# echo "input_file: ${input_file}"
# echo "freshness_flag: ${freshness_flag}"
# echo "cutoff_date: ${cutoff_date}"
# python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

# exp_name="brave_exp5"
# query_type="llm_rewrite"
# input_file="./mar13_pipeline_data.pickle"
# freshness_flag="True"
# cutoff_date="2026-03-01"

# echo "exp_name: ${exp_name}"
# echo "query_type: ${query_type}"
# echo "input_file: ${input_file}"
# echo "freshness_flag: ${freshness_flag}"
# echo "cutoff_date: ${cutoff_date}"
# python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

# exp_name="brave_exp6"
# query_type="llm_rewrite"
# input_file="./mar13_pipeline_data.pickle"
# freshness_flag="False"
# cutoff_date="2026-03-01"

# echo "exp_name: ${exp_name}"
# echo "query_type: ${query_type}"
# echo "input_file: ${input_file}"
# echo "freshness_flag: ${freshness_flag}"
# echo "cutoff_date: ${cutoff_date}"
# python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

exp_name="brave_exp7"
query_type="llm_rewrite"
input_file="./error_points.pickle"
freshness_flag="True"
cutoff_date="2026-03-01"

echo "exp_name: ${exp_name}"
echo "query_type: ${query_type}"
echo "input_file: ${input_file}"
echo "freshness_flag: ${freshness_flag}"
echo "cutoff_date: ${cutoff_date}"
python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}

exp_name="brave_exp8"
query_type="llm_rewrite"
input_file="./error_points.pickle"
freshness_flag="False"
cutoff_date="2026-03-01"

echo "exp_name: ${exp_name}"
echo "query_type: ${query_type}"
echo "input_file: ${input_file}"
echo "freshness_flag: ${freshness_flag}"
echo "cutoff_date: ${cutoff_date}"
python brave_api.py --input_path ${input_file} --query_type ${query_type} --exp_name ${exp_name} --FRESHNESS_FLAG ${freshness_flag} --cutoff_date ${cutoff_date}
