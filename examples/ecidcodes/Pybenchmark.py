import argparse
import time
import math
import csv
import os
from datetime import datetime
from ecidcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64

from collections import defaultdict


class DataFrame:
    def __init__(self):
        self.data = defaultdict(list)

    def push_back(self, key, value):
        self.data[key].append(value)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def save_to_csv(df_vector, filepath):
    if isinstance(df_vector, DataFrame):
        df_vector = [df_vector]  # Wrap the single DataFrame in a list

    if not df_vector:
        print("Error: DataFrame vector is empty.")
        return

    filename = f"{filepath.rstrip('/')}/{get_timestamp()}_benchmark.csv"

    unique_columns = set()
    for df in df_vector:
        unique_columns.update(df.data.keys())
    unique_columns = sorted(unique_columns)

    max_rows = max(
        (len(df.data.get(col, [])) for df in df_vector for col in unique_columns),
        default=0,
    )

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(unique_columns)

        for i in range(max_rows):
            row = []
            for col in unique_columns:
                value_written = False
                for df in df_vector:
                    if col in df.data and i < len(df.data[col]):
                        row.append(df.data[col][i])
                        value_written = True
                        break
                if not value_written:
                    row.append("")
            writer.writerow(row)

    print(f"Benchmark results saved to: {filename}")


def idcode_exec(
    Id,
    gf2_exp,
    data_type,
    input_data,
    encoder,
    tag_pos,
    tag_pos_in,
    rm_order,
    vec_len=0,
    n_iter=1,
):
    gf2_exp_inner = gf2_exp // 2
    gf_size = 2**gf2_exp
    gf_size_inner = 2**gf2_exp_inner
    col_name = f"{encoder}_{vec_len}_{gf2_exp}_{tag_pos}"

    message = []

    if data_type == "txt":
        with open(input_data, "r") as file:
            message = Id.read_inputfile_sequence(input_data, False)
    elif data_type == "bin":
        message = Id.read_file(input_data, gf2_exp)
    elif data_type == "random":
        if gf2_exp >= 33:
            vec_len_ = vec_len // 8
        elif gf2_exp >= 17:
            vec_len_ = vec_len // 4
        elif gf2_exp >= 9:
            vec_len_ = vec_len // 2
        else:
            vec_len_ = vec_len

        message = Id.generate_string_sequence(vec_len_)
        col_name = f"{encoder}_{vec_len_ * gf2_exp // 8}_{gf2_exp}_{tag_pos}"
    else:
        print("Invalid data type!")
        return

    if gf2_exp <= 16 and encoder in ["RSID", "RS2ID", "RMID"]:
        Id.generate_gf_outer(gf2_exp)
        if encoder == "RS2ID":
            Id.generate_gf_inner(gf2_exp)
        print(f"Galois Field GF(2^{gf2_exp}) Generated Successfully")
    elif gf2_exp_inner <= 16 and encoder == "RS2ID":
        Id.generate_gf_inner(gf2_exp)
        print(
            f"Galois Field GF for Inner RS2ID (2^{gf2_exp_inner}) Generated Successfully"
        )

    benchmark_data = DataFrame()
    exp_arr, log_arr = Id.get_exp_arr(), Id.get_log_arr()
    exp_arr_in, log_arr_in = Id.get_exp_arr_in(), Id.get_log_arr_in()
    for _ in range(n_iter):
        start_time = time.time_ns()
        if encoder == "RSID":
            print(
                f"Running RSID encoding with vec_len={vec_len}, tag_pos={tag_pos}, gf2_exp={gf2_exp}"
            )
            symbol_rs = Id.rsid(message, tag_pos, exp_arr, log_arr, gf2_exp)
        elif encoder == "RS2ID":
            print(
                f"Running RS2ID encoding with vec_len={vec_len}, tag_pos={tag_pos}, tag_pos_in={tag_pos_in}, gf2_exp={gf2_exp}"
            )
            symbol_rs2 = Id.rs2id(
                message,
                tag_pos,
                tag_pos_in,
                exp_arr,
                log_arr,
                exp_arr_in,
                log_arr_in,
                gf2_exp,
            )
        elif encoder == "RMID":
            print(
                f"Running RMID encoding with vec_len={vec_len}, tag_pos={tag_pos}, rm_order={rm_order}, gf2_exp={gf2_exp}"
            )
            symbol_rm = Id.rmid(message, tag_pos, rm_order, exp_arr, log_arr, gf2_exp)
        elif encoder == "SHA1ID":
            print(f"Running SHA1ID encoding with vec_len={vec_len}, gf2_exp={gf2_exp}")
            sha1id_code = Id.sha1id(message, gf2_exp)
        elif encoder == "SHA256ID":
            print(
                f"Running SHA256ID encoding with vec_len={vec_len}, gf2_exp={gf2_exp}"
            )
            sha256id_code = Id.sha256id(message, gf2_exp)
        else:
            print("Invalid encoder!")
            return
        end_time = time.time_ns()

        duration = end_time - start_time
        benchmark_data.push_back(col_name, duration)

    return benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Idcodes Library Main Function")
    parser.add_argument(
        "--vec_len",
        type=int,
        nargs="+",
        required=True,
        help="Vector length in Bytes for benchmarking, positive integer >0",
    )
    parser.add_argument(
        "--id_code_type",
        type=str,
        nargs="+",
        required=True,
        help="ID code type: SHA256ID | SHA1ID | RSID | RS2ID | RMID",
    )
    parser.add_argument(
        "--tag_pos",
        type=int,
        default=2,
        help="Tag Position for encoding: Positive integer >=0",
    )
    parser.add_argument(
        "--gf2_exp",
        type=int,
        nargs="+",
        required=True,
        help="GF(2^n) Exponent values, positive integer >=0 max 64",
    )
    parser.add_argument(
        "--data_type", type=str, default="txt", help="Data Type: txt | bin | random"
    )
    parser.add_argument(
        "--input_data", type=str, default="", help="Message Input File, full path"
    )
    parser.add_argument(
        "--rs2_inner_tag_pos",
        type=int,
        default=2,
        help="Concatenated RS Inner tag position, positive integer >=0",
    )
    parser.add_argument(
        "--rm_order",
        type=int,
        default=1,
        help="Reed Muller monomial order, positive integer >=0",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--n_iter", type=int, default=1, help="Number of iterations for benchmarking"
    )
    parser.add_argument(
        "--csv_file_path", type=str, default="./", help="Path to save CSV output"
    )

    args = parser.parse_args()

    valid_id_types = {"SHA256ID", "SHA1ID", "RSID", "RS2ID", "RMID"}
    valid_data_types = {"txt", "bin", "random"}

    for encoder in args.id_code_type:
        if encoder not in valid_id_types:
            raise ValueError(f"Invalid id_code_type: {encoder}")

    if args.data_type not in valid_data_types:
        raise ValueError(f"Invalid data_type: {args.data_type}")

    if not args.input_data and args.data_type != "random":
        raise ValueError("Input data is empty!")

    all_benchmark_data = DataFrame()
    for encoder in args.id_code_type:
        for gf2_exp in args.gf2_exp:
            if gf2_exp <= 8:
                Id = IDCODES_U8()
            elif gf2_exp <= 16:
                Id = IDCODES_U16()
            elif gf2_exp <= 32:
                Id = IDCODES_U32()
            elif gf2_exp <= 64:
                Id = IDCODES_U64()
            else:
                raise ValueError(
                    "Invalid GF(2^n) exponent value. Please enter a correct parameter!"
                )

            for vec_len in args.vec_len:
                benchmark_data = idcode_exec(
                    Id,
                    gf2_exp,
                    args.data_type,
                    args.input_data,
                    encoder,
                    args.tag_pos,
                    args.rs2_inner_tag_pos,
                    args.rm_order,
                    vec_len,
                    args.n_iter,
                )
                all_benchmark_data.data.update(benchmark_data.data)

    if args.csv_file_path:
        save_to_csv(all_benchmark_data, args.csv_file_path)


if __name__ == "__main__":
    main()
