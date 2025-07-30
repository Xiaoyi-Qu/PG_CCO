'''
File: createpbs.py
Author: Xiaoyi Qu
--------------------------------------------
Description: create pbs files for job submission
'''

# We will submit two pbs files.
# One is the small size,
# The other is the large size, i.e. m+n <= 4000(Temporary)

import os
import glob
from csv import DictReader

def create_in_one_file(task_name, node, vmem, pmem, queue,
           prob_list):
    contents = f'#PBS -N {task_name}\n'
    contents += f'#PBS -e /home/xiq322/PG_CCO/PG-General/test/{task_name}.err\n'
    contents += f'#PBS -o /home/xiq322/PG_CCO/PG-General/test/{task_name}.out\n'
    contents += f'#PBS -l nodes={node}:ppn=2\n'
    contents += f'#PBS -l vmem={vmem},pmem={pmem}\n'
    contents += f'#PBS -q {queue}\n'
    contents += f'#PBS -V\n\n'
    contents += 'ulimit -a\n\n'
    contents += f'cd /home/xiq322/PG_CCO/PG-General/test/experiements\n\n'
    set_of_commands = ""

    for prob in prob_list:
        command = f'/home/xiq322/miniconda3/bin/python main.py --name {prob["Problem Name"]} --kappav 0.02\n'
        # command = f'/home/xiq322/julia-1.8.3/bin/julia cutestprob.jl --name {prob["Name"]} --lambda 1000\n'
        set_of_commands += command
    contents += set_of_commands
    filename = f'./{task_name}.pbs'

    with open(filename, "w") as pbsfile:
        pbsfile.write(contents)

if __name__ == '__main__':
    # clean all existing pbs files
    # for f in glob.glob("*.pbs"):
    #     os.remove(f)

    # Handle the test_prob_small.csv file
    with open("./multipliers.csv", 'r') as f1:
        dict_reader1 = DictReader(f1)
        prob_list_small = list(dict_reader1)

    create_in_one_file('small', '1', '31gb', '31gb', 'long', prob_list_small)

    # create_in_one_file('big', '1', '31gb', '31gb', 'long', prob_list_big)