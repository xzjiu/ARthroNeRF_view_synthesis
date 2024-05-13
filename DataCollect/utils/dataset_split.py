import csv

def stream_read(file):
    stream_dict = []
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            stream_dict.append(row)
    return stream_dict

def data_split(tracker_file, ratio = 0.2):
    streams = stream_read(tracker_file)
    interval = 1//ratio
    train_data = [['ID', 'Image', 'Timestamps', 'Tracking']]
    test_data = [['ID', 'Image', 'Timestamps', 'Tracking']]
    for i, stream in enumerate(streams):
        if i % interval == 0:
            test_data.append([stream['ID'], stream['Image'],stream['Timestamps'],stream['Tracking']])
        else:
            train_data.append([stream['ID'], stream['Image'],stream['Timestamps'],stream['Tracking']])

    # Write and save image information
    with open('train_pivot.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)

    with open('test_pivot.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_data)

data_split('pivot.csv')