

import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./components/") # 添加模型组件路径
from applications.handpose_local_app import main_handpose_x

def demo_logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("             WELCOME      ")
    print("           << APP_X >>         ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")

if __name__ == '__main__':
    demo_logo()
    cfg_file = "./lib/hand_lib/cfg/handpose.cfg"
    main_handpose_x(cfg_file)#加载 handpose 应用

    print("well done ~")
