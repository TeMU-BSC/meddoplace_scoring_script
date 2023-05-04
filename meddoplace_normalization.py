from utils.geonorm_utils import geonames2coords_dict, checkPluscode, pluscode2coord, calculate_scores, print_stats, calculat_fscore, calculat_fscore_per_entity
from optparse import OptionParser
import sys
import os
import warnings
import pandas as pd
import numpy as np
import logging as log

general_path = os.getcwd().split(
    "meddoplace-evaluation-script")[0]+"meddoplace-evaluation-script/"
sys.path.append(general_path+'src/')

def main(argv=None):
    # load logger
    LOG_FILE = '/tmp/ADE_Eval.log'
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler(LOG_FILE)])

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input",
                      help="Input directory where the res and pred forlders are")
    parser.add_option("-o", "--output", dest="output",
                      help="output file where the result will be printed")
    parser.add_option("-d", "--dictionary", dest="dictionary",
                      help="file with term list to normalize, one per line")
    parser.add_option("-p", "--opcion", dest="opcion", help="TO BE DONE")
    # parser.add_option("-r", "--redact", dest="redact", help="do not write target sentences", default=None)
    (options, args) = parser.parse_args(argv)
    out_file = os.path.join(options.output, 'scores.txt')
    # Read gold_standard and predictions
    # get files in prediction zip file
    pred_dir = os.path.join(options.input, 'res')
    pred_files = [x for x in os.listdir(
        pred_dir) if not os.path.isdir(os.path.join(pred_dir, x))]
    pred_files = [x for x in pred_files if x[0] not in ["_", "."] and x[-4:] == '.tsv']
    print(pred_files)
    if not pred_files:
        log.error("No valid files found in archive. \
                  \nMake sure file names do not start with . or _ characters")
        sys.exit(0)
    if len(pred_files) > 1:
        for doc in pred_dir:
            log.error(str(doc))
        log.error("More than one valid files found in archive. \
                  \nMake sure only one valid file is available.")
        sys.exit(0)
    pred_file = os.path.join(options.input, "res/"+pred_files[0])
    # get files in gold zip file
    gold_dir = os.path.join(options.input, 'ref')
    gold_files = [x for x in os.listdir(
        gold_dir) if not os.path.isdir(os.path.join(gold_dir, x))]
    gold_files = [x for x in gold_files if x[0] not in ["_", "."]]
    if not gold_files:
        log.error("No valid files found in archive. \
                  \nMake sure file names do not start with . or _ characters")
        sys.exit(0)
    if len(gold_files) > 1:
        for doc in pred_dir:
            log.error(str(doc))
        log.error("More than one valid files found in archive. \
                  \nMake sure only one valid file is available.")
        sys.exit(0)
    gold_file = os.path.join(options.input, "ref/"+gold_files[0])
    print("Reading *.tsv files")
    df_gs = pd.read_csv(gold_file, sep="\t")
    df_gs = df_gs.drop_duplicates()
    df_preds = pd.read_csv(pred_file, sep="\t")
    df_preds = df_preds.drop_duplicates(
        subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)

    # We will use the recall value in these tasks as the formula is equivalent
    if options.opcion == "PC":
        calculate_PC(df_gs, df_preds, out_file)
    elif options.opcion == "GN":
        calculate_GN(df_gs, df_preds, out_file, options)
    elif options.opcion == "SCTID":
        calculate_SCTID(df_gs, df_preds, out_file)

    elif options.opcion == "task3":
        calculate_Task3(df_gs, df_preds, out_file)
    elif options.opcion == "task1":
        calculate_Task1(df_gs, df_preds, out_file)

    elif options.opcion == "all":
        task_1 = calculate_Task1(df_gs, df_preds, out_file, write=False)
        task_21 = calculate_GN(df_gs, df_preds, out_file, options, write=False)
        task_22 = calculate_PC(df_gs, df_preds, out_file, write=False)
        task_23 = calculate_SCTID(df_gs, df_preds, out_file, write=False)
        task_3 = calculate_Task3(df_gs, df_preds, out_file, write=False)
        # print(task_1, task_21, task_22, task_23, task_3)
        F_score_AVG = (task_21["f_score"]+task_1["f_score"] +
                       task_22["f_score"]+task_23["f_score"]+task_3["f_score"])/5
        out = open(out_file, 'w')
        out.write("F_score_T1:"+str(task_1["f_score"])+"\n")
        out.write("F_score_T2-1:"+str(task_21["f_score"])+"\n")
        out.write("F_score_T2-2:"+str(task_22["f_score"])+"\n")
        out.write("F_score_T2-3:"+str(task_23["f_score"])+"\n")
        out.write("F_score_T3:"+str(task_3["f_score"])+"\n")
        out.write("F_score_AVG:"+str(F_score_AVG))
        out.flush()

    else:
        print(options.opcion)
        print("Incorrect option, please choose from: task1, GN, PC, SCTID, task3, all.")


def calculate_Task1(df_gs, df_preds, out_file, write=True):
    print("Computing evaluation scores for Task 1")
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text",  "label"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label"]].values.tolist()).to_list()
    result = calculat_fscore_per_entity(
        list_gs_per_doc, list_preds_per_doc, strict=False)

    if write:
        out = open(out_file, 'w')
        for k, v in result.items():
            out.write(str(k)+":"+str(v)+"\n")
    else:
        return result


def calculate_Task3(df_gs, df_preds, out_file, write=True):
    print("Computing evaluation scores for Task 3")
    df_gs_sct = df_gs[df_gs["class"] != "None"].reset_index(drop=True)
    df_preds_sct = df_preds[(df_preds["class"] == 'ATENCION') | (df_preds["class"] == 'RESIDENCIA') |
                            (df_preds["class"] == 'MOVIMIENTO') | (df_preds["class"] == 'LUGAR-NATAL') |
                            (df_preds["class"] == 'OTHER')].reset_index(drop=True)
    list_gs_per_doc = df_gs_sct.groupby('filename').apply(lambda x: x[[
        'start_span', 'end_span', 'label', "filename", "class"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds_sct.groupby('filename').apply(
        lambda x: x[['start_span', 'end_span', "label", "filename", "class"]].values.tolist()).to_list()
    result = calculat_fscore_per_entity(
        list_gs_per_doc, list_preds_per_doc)
    if write:
        out = open(out_file, 'w')
        for k in result.keys():
            if k.endswith('recall'):
                out.write(str(k.replace("recall", "Accuracy"))+":"+str(result[k])+"\n")
    else:
        return result


def calculate_PC(df_gs, df_preds, out_file, write=True):
    print("Computing evaluation scores for pluscodes")
    # Filter pluscodes mentiones
    df_gs_pc = df_gs[(df_gs.source == "PC")].reset_index(drop=True)

    df_preds_pc = df_preds[df_preds.source == "PC"].reset_index(drop=True)
    # Check if pluscodes are correct
    df_preds_pc["isValid"] = df_preds_pc.normalization.apply(
        lambda x: checkPluscode(x))
    if df_preds_pc.shape[0] == np.sum(df_preds_pc.isValid.to_list()):
        print("All pluscodes are correctly formatted")
    else:

        log.error("Some pluscodes are not correctly formatted")
        for n in df_preds_pc[df_preds_pc["isValid"] == False].normalization:
            log.error("normalization not valid = "+n)
        # sys.exit(0)

    # Transform pluscodes to coordinates
    if write:
        df_gs_pc["coords"] = df_gs_pc.normalization.apply(
            lambda x: pluscode2coord(x))
        df_preds_pc["coords"] = df_preds_pc.normalization.apply(
            lambda x: pluscode2coord(x))
    else:
        df_gs_pc["coords"] = 0
        df_preds_pc["coords"] = 0
    # Group predictions per document (one list per document)
    list_gs_per_doc = df_gs_pc.groupby('filename').apply(lambda x: x[[
        'coords', 'start_span', 'end_span', "filename", "normalization"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds_pc.groupby('filename').apply(lambda x: x[[
        'coords', 'start_span', 'end_span', "filename", "normalization"]].values.tolist()).to_list()

    output = calculate_scores(
        list_preds_per_doc, list_gs_per_doc, inspect=True, task="PC", write=write)
    scores = print_stats(accuracy=list(
        output['accuracy'].values()), scores=output['f_score'], plot=True, write=write)
    if write:
        out = open(out_file, 'w')
        out.write("accuracy_at_161km:{}\nAUC:{}\nMedian_error:{}\nMean_error:{}\nAccuracy:{}\n".format(
            scores["accuracy_at_161"], scores["auc"], scores["median"], scores["mean"], scores["recall"]))
        out.flush()
    else:
        return scores


def calculate_GN(df_gs, df_preds, out_file, options, write=True):
    print("Computing evaluation scores for geonames")
    # Filter geonames mentiones
    df_gs_geo = df_gs[df_gs.source == "GN"].reset_index(drop=True)
    df_preds_geo = df_preds[df_preds.source == "GN"].reset_index(drop=True)
    # Ensure normalization is int value

    df_gs_geo["normalization"] = df_gs_geo.normalization.apply(
        lambda x: int(x) if str(x).isdigit() else x)
    df_preds_geo["normalization"] = df_preds_geo.normalization.apply(
        lambda x: int(x) if str(x).isdigit() else x)

    # Read diccionario
    if write:
        print("Reading geonames dictionary to transform ids to coordinates...")
        geoname2coord = geonames2coords_dict(options.dictionary)

    # Check if geocodes are correct
        df_gs_geo["coords"] = df_gs_geo.normalization.map(geoname2coord)
        df_preds_geo["coords"] = df_preds_geo.normalization.map(
            geoname2coord)
    else:
        df_gs_geo["coords"] = df_gs_geo["normalization"]
        df_preds_geo["coords"] = df_preds_geo["normalization"]
    # Group predictions per document (one list per document)
    list_gs_per_doc = df_gs_geo.groupby('filename').apply(
        lambda x: x[['coords', 'start_span', 'end_span', "filename"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds_geo.groupby('filename').apply(
        lambda x: x[['coords', 'start_span', 'end_span', "filename"]].values.tolist()).to_list()
    # Compute scores
    output = calculate_scores(
        list_preds_per_doc, list_gs_per_doc, inspect=True, write=write)

    scores = print_stats(accuracy=list(
        output['accuracy'].values()), scores=output['f_score'], plot=True, write=write)

    # Print results
    if write:
        out = open(out_file, 'w')
        # out.write("######################GEONAMES NORMALIZATION SCORES######################\n")
        out.write("accuracy_at_161:{}\nAUC:{}\nMedian_error:{}\nMean_error:{}\nAccuracy:{}\n".format(
            scores["accuracy_at_161"], scores["auc"], scores["median"], scores["mean"], scores["recall"]))
        # Lower AUC is better value
        out.flush()
    else:
        return scores


def calculate_SCTID(df_gs, df_preds, out_file, write=True):
    print("Computing evaluation scores for snomed-ct")
    # Filter geonames mentiones
    df_gs_sct = df_gs[df_gs.source == "SCTID"].reset_index(drop=True)
    df_preds_sct = df_preds[df_preds.source == "SCTID"].reset_index(drop=True)
    list_gs_per_doc = df_gs_sct.groupby('filename').apply(lambda x: x[[
        'start_span', 'end_span', 'label', "filename", "normalization"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds_sct.groupby('filename').apply(
        lambda x: x[['start_span', 'end_span', "label", "filename", "normalization"]].values.tolist()).to_list()
    result = calculat_fscore(list_gs_per_doc, list_preds_per_doc)
    if write:
        out = open(out_file, 'w')
        out.write("Accuracy:{}\n".format(result["recall"]))
    else:
        return result


if __name__ == "__main__":
    sys.exit(main())
