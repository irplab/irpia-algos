import argparse
import json
from tasks import predict_domain


def parse_arguments():
    parser = argparse.ArgumentParser(description='Irpia module for domain prediction from title.')
    parser.add_argument('--title', dest='title', default='',
                        help='Metadata title')
    parser.add_argument('--text', dest='text', default='',
                        help='Metadata text')
    parser.add_argument('--url', dest='url', default='',
                        help='Metadata url')
    args, unknown = parser.parse_known_args()
    return args


def is_not_blank(s):
    return bool(s and not s.isspace())


if __name__ == '__main__':
    args = parse_arguments()
    if is_not_blank(args.title):
        result = predict_domain.delay(title=args.title, description=args.text)
        print(result.get(timeout=3))
    else:
        print(json.dumps({'domain': []}))
