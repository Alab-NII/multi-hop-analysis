# -*- coding: utf-8 -*-
import gzip
import json
import os
import pickle

import joblib


def make_dirs(dirname):
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def read_file(filename):
    with open(filename, mode="rb") as f:
        return f.read()


def write_file(data, filename):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="wb") as f:
        f.write(data)


def read_text(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        return f.read()


def write_text(text, filename, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        f.write(text)


def read_lines(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            f.write(linesep)
            f.flush()


def read_json(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        return json.load(f)


def write_json(obj, filename, indent=None, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        json.dump(obj, fp=f, ensure_ascii=False, indent=indent)


def read_jsonlines(filename, encoding="UTF-8"):
    for json_line in read_lines(filename, encoding=encoding):
        yield json.loads(json_line)


def write_jsonlines(objs, filename, linesep="\n", encoding="UTF-8"):
    json_lines = (json.dumps(obj, ensure_ascii=False) for obj in objs)
    write_lines(json_lines, filename=filename, linesep=linesep, encoding=encoding)


def deserialize_object(filename, mmap_mode=None):
    return joblib.load(filename, mmap_mode=mmap_mode)


def serialize_object(obj, filename, compress=3, protocol=None, cache_size=None):
    make_dirs(os.path.dirname(filename))

    joblib.dump(
        obj,
        filename=filename,
        compress=compress,
        protocol=protocol,
        cache_size=cache_size,
    )


def deserialize_objects(filename):
    with gzip.open(filename, mode="rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def serialize_objects(objs, filename):
    make_dirs(os.path.dirname(filename))

    with gzip.open(filename, mode="wb") as f:
        for obj in objs:
            pickle.dump(obj, f)
