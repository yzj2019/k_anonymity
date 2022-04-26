import argparse


def parse_args():
    """
    解析命令行参数
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    # 后面的help是描述
    parser.add_argument('path', help="The path of the raw data file (*.data)", type=str)
    parser.add_argument('k', help='The value of k for k_anonymity', type=int)
    parser.add_argument('-ms', help='The value of max suppression, you have to adjust it if you use samarati', type=int, default=10)
    args = parser.parse_args()                                       # 步骤四          
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f'the path is {args.path}, type is {type(args.path)}')
    print(f'the k value is {args.k}, type is {type(args.k)}')
    print(f'the max suppression value is {args.ms}, type is {type(args.ms)}')