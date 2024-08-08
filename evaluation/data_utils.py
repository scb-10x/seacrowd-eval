from seacrowd import SEACrowdConfigHelper
from seacrowd.utils.constants import Tasks
import pandas as pd
import datasets
from enum import Enum
import datasets

def patch_resolve_trust_remote_code():
    def resolve_trust_remote_code(trust_remote_code: bool | None, repo_id: str):
        print('Patch `trust_remote_code` to enable fully auto-run. Beware of the risk of code injection in the dataset.')
        return True
    datasets.load.resolve_trust_remote_code = resolve_trust_remote_code

patch_resolve_trust_remote_code()


NLU_TASK_LIST = {
    "wisesight_thai_sentiment_seacrowd_text",
    "m3exam_tha_seacrowd_qa",
    "xcopa_tha_seacrowd_qa",
    "belebele_tha_thai_seacrowd_qa",
    "xnli.tha_seacrowd_pairs",
    'thaiexam_qa'
}


NLG_TASK_LIST = [
    "xl_sum_tha_seacrowd_t2t",
    "flores200_eng_Latn_tha_Thai_seacrowd_t2t",
    "flores200_tha_Thai_eng_Latn_seacrowd_t2t",
    "iapp_squad_seacrowd_qa",
]



def load_nlu_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in NLU_TASK_LIST:
        if config_name == 'thaiexam_qa':
            ds = datasets.load_dataset('kunato/thai-exam-seacrowd')
            cfg_name_to_dset_map[config_name] = (ds, Tasks.COMMONSENSE_REASONING)
        else:
            schema = config_name.split('_')[-1]
            con = nc_conhelp.for_config_name(config_name)
            cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])
    return cfg_name_to_dset_map


def load_nlg_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in NLG_TASK_LIST:
        schema = config_name.split('_')[-1]
        con = nc_conhelp.for_config_name(config_name)
        cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])
    return cfg_name_to_dset_map