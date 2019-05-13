from rasa.nlu import training_data
from rasa.nlu.utils import write_to_file


def convert_training_data(data_file, out_file, output_format, language):
    td = training_data.load_data(data_file, language)

    if output_format == "md":
        output = td.as_markdown()
    else:
        output = td.as_json(indent=2)

    write_to_file(out_file, output)


def main(args):
    convert_training_data(args.data_file, args.out_file, args.format, args.language)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.convert` directly is "
        "no longer supported. "
        "Please use `rasa data` instead."
    )
