import glob
import io
import multiprocessing
import os.path
import pathlib

import ujson
import zstandard
from tqdm.contrib.concurrent import process_map

__HERE__ = pathlib.Path(__file__).parent.resolve()


def jsonl_extract_transform_write(infile, outfile, func, max_lines=-1):
    line_counter = 0
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(infile) as reader:
        cctx = zstandard.ZstdCompressor()
        with cctx.stream_writer(outfile, closefd=False) as writer:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                func_output = func(line)
                writer.write(f'{func_output}\n'.encode())
                line_counter += 1
                if -1 < max_lines <= line_counter:
                    print(f'Stopping after {line_counter} lines')
                    break
    return line_counter


def pile_get_text(line):
    # Just getting rid of the meta tag for a consistent schema
    json_obj = ujson.loads(line)
    text = json_obj['text']
    text_json = ujson.dumps({'text': text})
    return text_json


def create_pile_domain_mix_mapper(args):
    domain_data_file_path, output_file_path = args
    with open(output_file_path, 'wb+') as outfile:
        with open(domain_data_file_path, 'rb') as infile_d:
            num_domain_samples = jsonl_extract_transform_write(infile_d, outfile, pile_get_text)
            return num_domain_samples


def create_pile_domain_mix(domain_data_file_path: str,
                           pile_file_path: str,
                           output_dir: str,
                           max_files: int = -1,
                           max_workers: int = multiprocessing.cpu_count()):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise IOError('Output path already exists')

    domain_data_file_path_expanded = glob.glob(domain_data_file_path)
    if max_files > 0:
        print(f'Using {max_files} domain data files')
        domain_data_file_path_expanded = domain_data_file_path_expanded[:max_files]

    domain_data_processed_paths = [os.path.join(output_dir, os.path.basename(x))
                                   for x in domain_data_file_path_expanded]
    print('Processing domain data samples')
    file_sample_counts = process_map(create_pile_domain_mix_mapper,
                                     zip(domain_data_file_path_expanded, domain_data_processed_paths),
                                     max_workers=max_workers)

    num_domain_samples = sum(file_sample_counts)
    print(f'Number of domain samples: {num_domain_samples}')

    print('Processing Pile data samples')  # Find a way to paralellize this, as the Pile files are 15gb each.
    pile_output_path = os.path.join(output_dir, 'pile.jsonl.zst')
    with open(pile_output_path, 'wb+') as outfile:
        with open(pile_file_path, 'rb') as infile_p:
            jsonl_extract_transform_write(infile_p, outfile, pile_get_text, num_domain_samples)


def read_pile_texts(input_file_path):
    """
    Reads a Pile dataset file in zstd-compressed JSON format and returns a list of 'text' fields.

    :param input_file_path: The path to the input file.
    :type input_file_path: str
    :return: A list of 'text' fields from each line of the input file.
    :rtype: List[str]
    :raises FileNotFoundError: If the input file path does not exist.
    :raises ValueError: If the input file path is not a string.
    :example: read_pile_texts('pile_texts.zst')
    """
    with open(input_file_path, 'rb') as infile:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(infile) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            return [ujson.loads(line)['text'] for line in text_stream]


if __name__ == '__main__':
    repo_root = __HERE__.parent.parent
    domain_data_file_path = str(repo_root / 'data/pile_uspto/*.jsonl.zst')
    pile_file_path = str(repo_root / 'data/pile_01/01.jsonl.zst')
    output_dir = str(repo_root / 'data/mix_uspto_all')

    create_pile_domain_mix(domain_data_file_path, pile_file_path, output_dir, max_files=2, max_workers=4)
    # print(read_pile_texts(str(output_dir))[0])
