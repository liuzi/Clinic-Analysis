import os 
from utils.tools import *
from os.path import *

def run_clamp(
    input_path = "/data/liu/mimic3/CLAMP_NER/input", 
    output_path = "/data/liu/mimic3/CLAMP_NER/ner-attribute/output"):

    dirs = [dir for dir in os.listdir(input_path) if os.path.splitext(dir)[0].startswith('Rule')]
    print(dirs)
    nohup_cmd = "cd /home/liu/package/ClampCmd_1.6.1 && nohup ./run_keysession_nerattr_pipline_v2.sh ner-attribute %s \"%s\" >> \"/home/liu/nohup/%s_%s_nohup.out\" 2>&1 &"
    
    for dir in dirs:
        input_root = join(input_path, dir)
        output_root = join(output_path, dir)
        # print(output_root)
        create_folder_overwrite(output_root)
        sub_dirs = [join(output_root, subdir) for subdir in os.listdir(input_root)]
        [create_folder_overwrite(subdir) for subdir in sub_dirs]
        [os.system(nohup_cmd%((dir, sub_dir)*2)) for sub_dir in os.listdir(input_root)]
        # print([sub_dirs])
        

run_clamp()
# section="HOSPITAL_COURSE"
# rule_index=0
# rule="Rule%d"%(rule_index)
# os.system(nohup_cmd%((rule, section,)*2))

