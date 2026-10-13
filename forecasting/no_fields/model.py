"""本地实现：Full 的 field_forecast=False 对照，仍保留原历史分解和损失。"""
import copy


def configure(config):
    result = copy.deepcopy(config)
    result['model']['field_forecast'] = False
    return result
