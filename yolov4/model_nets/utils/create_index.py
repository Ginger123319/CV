import struct
import typing
import tfrecord
import numpy as np


typename_mapping = {
            "byte": "bytes_list",
            "float": "float_list",
            "int": "int64_list"
        }


def _parse_record(record, description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None):
    example = tfrecord.example_pb2.Example()
    example.ParseFromString(record)

    all_keys = list(example.features.feature.keys())
    if description is None:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")
        # NOTE: We assume that each key in the example has only one field
        # (either "bytes_list", "float_list", or "int64_list")!
        field = example.features.feature[key].ListFields()[0]
        inferred_typename, value = field[0].name, field[1].value
        if typename is not None:
            tf_typename = typename_mapping[typename]
            if tf_typename != inferred_typename:
                reversed_mapping = {v: k for k, v in typename_mapping.items()}
                raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                                f"(should be '{reversed_mapping[inferred_typename]}').")

        # Decode raw bytes into respective data types
        if inferred_typename == "bytes_list":
            value = np.frombuffer(value[0], dtype=np.uint8)
        elif inferred_typename == "float_list":
            value = np.array(value, dtype=np.float32)
        elif inferred_typename == "int64_list":
            value = np.array(value, dtype=np.int32)
        features[key] = value
    return features


def create_index(tfrecord_file: str, index_file: str) -> None:

    """Create index from the tfrecords file.

    Stores starting location (byte) and length (in bytes) of each
    serialized record.

    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.

    index_file: str
        Path where to store the index file.
    """
    infile = open(tfrecord_file, "rb")
    outfile = open(index_file, "w")

    while True:
        current = infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            record = infile.read(proto_len)
            infile.read(4)

            path = bytes(_parse_record(record)['path'])
            path = path.decode('utf-8')
            # path = path.replace('D:\\Dataset\\cv\\fruits\\data\\image\\', '/opt/aps/dataset/8b4bd055-4a00-4f64-9fd0-9215352d6698/data/data/image/') # 
            
            outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
        except:
            print("Failed to parse TFRecord.")
            break
    infile.close()
    outfile.close()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data2.csv")

    create_index("0_data_240.tfrecord", "a.index", df)
