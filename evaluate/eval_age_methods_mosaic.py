import argparse

datasets_available = ["vggface2", "lfw", "lap", "imdbwiki"]
partitions_available = ["train", "val", "validation", "test"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", type=str, required=True, choices=datasets_available, help="Dataset on which to test")
parser.add_argument("--partition", dest="partition", type=str, default="test", choices=partitions_available, help="Partition on which to test")
parser.add_argument("--path", dest="path", type=str, required=True, help="Path of nets")
parser.add_argument("--out_path", dest="out_path", type=str, default="results", help="Directory into which to store results") 
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="Batch size")
parser.add_argument('--gpu', dest="gpu", type=str, default=None, help="Gpu to use")
parser.add_argument("--avoid_roi", action="store_true", help="No roi will be focused")
args = parser.parse_args()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from evaluate_utils import *

CSV_FILE_NAME = "results_{}_{}_of{}.csv"
MOSAIC_FILE_NAME = "mosaic_{}_{}_of{}.jpg"
CHART_FILE_NAME = "chart_{}_{}_of{}.png"
SUMMARY_FILE_NAME = "summary_{}_{}_of{}.txt"

categorical_classes = [i for i in range(0, 101)]


def _refactor_data(data):
    restored = list()
    for item in data:
        if item[3] is not None:
            roi = [np.int(x) for x in item[3:7]]
            item  = [item[0], np.float(item[1]), np.float(item[2]), roi]
        restored.append(item)
    return restored


def total_mse(data, index_pred, index_true):
    mse = None
    for item in data:
        square_error = (float(item[index_pred])-float(item[index_true]))**2
        mse = square_error if mse is None else mse+square_error
    return mse/len(data) if mse is not None else None


def total_mae(data, index_pred, index_true):
    mae = None
    for item in data:
        error = abs(item[index_pred]-item[index_true])
        mae = error if mae is None else mae+error
    return mae/len(data) if mae is not None else None


def total_me(data, index_pred, index_true):
    mean_error = None
    for item in data:
        error = item[index_pred]-item[index_true]
        mean_error = error if mean_error is None else mean_error+error
    return mean_error/len(data) if mean_error is not None else None


def get_class(data_item):
    # TODO
    if data_item in categorical_classes:
        return data_item
    else:
        raise Exception("Class error")


def total_class_me(data, index_pred, index_true):
    classes_error = defaultdict(list)
    maximum, minimum = None, None
    for item in data:
        item_true = item[index_true]
        error = item[index_pred] - item_true
        
        if maximum is None or maximum < item_true:
            maximum = item_true
        
        if minimum is None or minimum > item_true:
            minimum = item_true

        classes_error[get_class(int(np.round(item_true)))].append(error)

    classes_error_avg = dict()
    for i in categorical_classes:
        if classes_error[i]:
            classes_error_avg[i] = sum(classes_error[i])/len(classes_error[i])
        else:
            classes_error_avg[i] = None

    print("----- Minimum:", minimum)
    print("+++++ Maximum:", maximum)
        
    return classes_error_avg


def total_error(data, index_pred, index_true, thresholds):
    errors = dict()
    for th in thresholds:
        counter = 0
        for item in data:
            error = abs(item[index_pred]-item[index_true])
            if error >= th:
                counter += 1
        errors[th] = counter*100/len(data)
    return errors


def run_test(modelpath, batch_size=64, partition='test'):
    model, INPUT_SHAPE = load_keras_model(modelpath)
    dataset = Dataset(partition=partition, target_shape=INPUT_SHAPE, augment=False, preprocessing='vggface2')
    data_gen = dataset.get_generator(batch_size, fullinfo=True)
    original_labels = list()
    image_paths = list()
    image_rois = list()
    predictions = list()
    for batch in tqdm(data_gen):
        predictions.extend(model.predict(batch[0]))
        original_labels.extend(batch[1])
        image_paths.extend(batch[2])
        image_rois.extend(batch[3])
    assert (len(image_paths) == len(predictions) == len(original_labels) == len(image_rois)), "Invalid prediction on batch"
    return image_paths, predictions, original_labels, image_rois


def run_all(path, run_test, batch_size, partition, dataset):
    all_results = dict()
    for p in get_allchecks(path):
        results_out_path = CSV_FILE_NAME.format(dataset, partition, p.split('/')[-2])
        if os.path.exists(os.path.join(args.out_path, results_out_path)):
            reference = _refactor_data(readcsv(os.path.join(args.out_path, results_out_path)))
        else:
            image_paths, predictions, original_labels, image_rois  = run_test(p, batch_size, partition)
            reference = zip_reference(image_paths, predictions, original_labels, image_rois)
            writecsv(os.path.join(args.out_path, results_out_path), reference)
        all_results[p] = reference
    return all_results

def zip_reference(image_paths, predictions, original_labels, image_rois):
    reference = list()
    for path, pred, original, roi in zip(image_paths, predictions, original_labels, image_rois):
        reference.append((path, pred[0], original, roi))
    return reference

def log_mse_mae(title, mse, mae):
    return "{}\n\tMean square error: {}\n\tMean absolute error: {}".format(title, mse, mae)

def log_mse_mae_me(title, mse, mae, me):
    return "{}\n\tMean square error: {}\n\tMean absolute error: {}\n\tMean error: {}".format(title, mse, mae, me)

def log_classes(title, data):
    col_labels = ["Age"] + ["+{}".format(i) for i in range(0, 10)]
    row_labels = ["{}s".format(i) for i in range(0,11)]
    values = []
    for i in range(int(len(categorical_classes)/10)):
        tmp = [row_labels[i]] + [data[category] for category in categorical_classes[i*10: (i+1)*10]]
        values.append(tmp)
    return "{}\n{}".format(title, tabulate(values, headers=col_labels, tablefmt="grid"))

def log_eps_score(eps):
    return "\n\tEps-score: {}".format(eps)

def create_mosaic(reference, out_path, size=(4,5), avoid_roi=False, images_root=None):
    rows, columns = size    
    mosaic = None
    extra_dim = np.array(reference).shape[1]
    mosaic_items = np.array(random.choices(reference, k=rows*columns))
    mosaic_items = np.reshape(mosaic_items, (rows, columns, extra_dim))
    mse = None
    mae = None

    for i in range(rows):
        mosaic_row = None
        for j in range(columns):
            mosaic_item = mosaic_items[i][j]

            image_path = mosaic_item[0] if images_root is None else os.path.join(images_root, mosaic_item[0])
            image_value = np.round(np.float(mosaic_item[1]), decimals=1)
            image_original_value = np.round(np.float(mosaic_item[2]), decimals=1) if mosaic_item[2] is not None else None
            image_roi = mosaic_item[3] if mosaic_item[3] is not None else None
            
            image = cv2.imread(image_path)
            assert image is not None, "Error loading image {}".format(image_path)

            if not (image_roi is None or avoid_roi):
                image = cut(image, image_roi)

            image = cv2.resize(image, (224, 224))
            cv2.rectangle(image,(0,0),(90,35),(0,0,0),cv2.FILLED)
            cv2.putText(image,str(image_value),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)

            if image_original_value is not None:
                cv2.rectangle(image,(91,0),(180,35),(0,0,255),cv2.FILLED)
                cv2.putText(image,str(image_original_value),(100,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
                square_error = (image_original_value-image_value)**2
                mse = square_error if mse is None else mse+square_error
                absolute_error = abs(image_original_value-image_value)
                mae = absolute_error if mae is None else mae+absolute_error

            mosaic_row = image if mosaic_row is None else np.concatenate((mosaic_row, image), axis=1)
        mosaic = mosaic_row if mosaic is None else np.concatenate((mosaic,mosaic_row),axis=0)
    cv2.imwrite(out_path, mosaic)
    mse = np.round(mse/(rows*columns), decimals=3) if mse is not None else None
    mae = np.round(mae/(rows*columns), decimals=3) if mae is not None else None
    return mse, mae


def create_error_chart(data, save_file_path, title=''):
    nbars = len(data)
    _ , ax = plt.subplots()
    ax.set_title(title)

    labels = [k for k in data.keys()]
    values = [v for v in data.values()]

    _ = ax.bar(np.arange(nbars), values, align='center', alpha=0.5)
    # print(data)
    ax.set_xticks(np.arange(0, nbars, step=1))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 101, step=10))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error percentage')
    plt.savefig(save_file_path, bbox_inches="tight", dpi=300)

def eps_score(predicted, mean, std):
    up = (predicted-mean)**2
    down = 2*std**2 
    if down == 0: down = np.finfo(np.float32).eps
    eps =  1-np.exp(-(up/down))
    return eps
    
def total_eps(reference, partition="test"):
    from chalearn_lap_appa_real_age import structured_lap_data_wrapper
    data = structured_lap_data_wrapper(partition)
    results = list()

    for (image_path, predicted, _, _) in reference:
        image_name = os.path.split(image_path)[-1]
        mean = float(data[image_name]['apparent_age'])
        std = float(data[image_name]['apparent_std'])
        results.append(eps_score(predicted, mean, std))

    return np.average(results)


if __name__ == '__main__':
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)

    if args.gpu is not None:
        select_gpu(args.gpu)

    partition = "val" if args.partition.startswith("val") else args.partition

    if args.dataset == "vggface2":
        from vgg2_dataset_age import Vgg2DatasetAge as Dataset
    elif args.dataset == "lfw":
        from lfw_dataset_age import LFWPlusDatasetAge as Dataset
    elif args.dataset == "lap":
        from chalearn_lap_appa_real_age import LAPAge as Dataset
    elif args.dataset == "imdbwiki":
        from imdbwiki_dataset_age import IMDBWIKIAge as Dataset
    
    if args.path.endswith('.hdf5'):
        results_out_path = CSV_FILE_NAME.format(args.dataset, partition, args.path.split('/')[-2])
        if os.path.exists(os.path.join(args.out_path, results_out_path)):
            reference = _refactor_data(readcsv(os.path.join(args.out_path, results_out_path)))
        else:
            image_paths, predictions, original_labels, image_rois  = run_test(args.path, args.batch_size, partition)
            reference = zip_reference(image_paths, predictions, original_labels, image_rois)
            writecsv(os.path.join(args.out_path, results_out_path), reference)         
        mosaic_out_path = MOSAIC_FILE_NAME.format(args.dataset, partition, args.path.split('/')[-2])
        mse, mae = create_mosaic(reference=reference, out_path=os.path.join(args.out_path, mosaic_out_path), avoid_roi=args.avoid_roi)

        reference_mse = total_mse(reference, index_pred=1, index_true=2)
        reference_mae = total_mae(reference, index_pred=1, index_true=2)
        reference_me = total_me(reference, index_pred=1, index_true=2)
        reference_classes_me = total_class_me(reference, index_pred=1, index_true=2)

        print("Network:", args.path.split('/')[-2])
        print("Dataset:", args.dataset, "- Partition:", partition)
        log_data_general = log_mse_mae_me("Entire dataset partition results:", reference_mse, reference_mae, reference_me)
        log_data_classes_general = log_classes("Entire dataset mean error over classes:", reference_classes_me)
        log_data_mosaic = log_mse_mae("Mosaic data:", mse, mae)
        
        if args.dataset == "lap":
            print("Calculating eps-score...")
            reference_eps_score = total_eps(reference)
            log_data_general += log_eps_score(reference_eps_score)
        
        summary_file = SUMMARY_FILE_NAME.format(args.dataset, partition, args.path.split('/')[-2])
        with open(os.path.join(args.out_path, summary_file), "w") as f:
            f.writelines([i+"\n\n" for i in [log_data_general, log_data_classes_general, log_data_mosaic]])
        print(log_data_general)
        print(log_data_classes_general)
        print()
        print(log_data_mosaic)

        # Creating error chart (inverted cs)
        reference_error = total_error(reference, index_pred=1, index_true=2, thresholds=[t for t in range(1, 11)])
        chart_out_path = CHART_FILE_NAME.format(args.dataset, partition, args.path.split('/')[-2])
        title_chart = "{} dataset. Percentage of errors depending on thresholds.".format(args.dataset.upper())
        create_error_chart(reference_error, os.path.join(args.out_path, chart_out_path), title_chart)

    else:
        # TODO else test
        results = run_all(args.path, run_test, args.batch_size, partition, args.dataset)
        for model_path, reference in results.items():   
            mosaic_out_path = MOSAIC_FILE_NAME.format(args.dataset, partition, model_path.split('/')[-2])
            mse, mae = create_mosaic(reference=reference, out_path=os.path.join(args.out_path, mosaic_out_path), avoid_roi=args.avoid_roi)

            reference_mse = total_mse(reference, index_pred=1, index_true=2)
            reference_mae = total_mae(reference, index_pred=1, index_true=2)
            reference_me = total_me(reference, index_pred=1, index_true=2)
            reference_classes_me = total_class_me(reference, index_pred=1, index_true=2)

            print("Network", model_path.split('/')[-2])
            print("Dataset:", args.dataset, "- Partition:", partition)
            log_data_general = log_mse_mae_me("Entire dataset partition results:", reference_mse, reference_mae, reference_me)
            log_data_classes_general = log_classes("Entire dataset mean error over classes:", reference_classes_me)
            log_data_mosaic = log_mse_mae("Mosaic data:", mse, mae)

            if args.dataset == "lap":
                print("Calculating eps-score...")
                reference_eps_score = total_eps(reference)
                log_data_general += log_eps_score(reference_eps_score)

            summary_file = SUMMARY_FILE_NAME.format(args.dataset, partition, model_path.split('/')[-2])
            with open(os.path.join(args.out_path, summary_file), "w") as f:
                f.writelines([i+"\n\n" for i in [log_data_general, log_data_classes_general, log_data_mosaic]])
            print(log_data_general)
            print(log_data_classes_general)
            print()
            print(log_data_mosaic)

            # Creating error chart (inverted cs)
            reference_error = total_error(reference, index_pred=1, index_true=2, thresholds=[t for t in range(1, 11)])
            chart_out_path = CHART_FILE_NAME.format(args.dataset, partition, model_path.split('/')[-2])
            title_chart = "{} dataset. Percentage of errors depending on thresholds.".format(args.dataset.upper())
            create_error_chart(reference_error, os.path.join(args.out_path, chart_out_path), title_chart)
            



    
