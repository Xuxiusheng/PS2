import json
import argparse

def modifyJson(args):
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    print("Before updating the JSON file:")
    print(data)
    
    fields = args.fields.split(' ')
    values = args.values.split(' ')
    assert len(fields) == len(values), "Number of fields and values must match."

    map_values = {k: v for k, v in zip(fields, values)}

    for k in map_values.keys():
        if k in data["val"][0]:
            data["val"][0][k] = map_values[k]
    
    print("After updating the JSON file:")
    print(data)

    with open(args.json_file, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update a JSON field value.')
    parser.add_argument('-j', '--json-file', type=str, help='Path to the JSON file.')
    parser.add_argument('-f', '--fields', type=str, help='The field to update in the JSON file.')
    parser.add_argument('-v', '--values', type=str, help='The new value for the specified field.')

    args = parser.parse_args()
    modifyJson(args)
