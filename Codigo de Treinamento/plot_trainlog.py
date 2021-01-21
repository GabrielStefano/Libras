import csv
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(8, 6))
def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)
        accuracies = []
        val_accuracies = []
        top_5_accuracies = []
        top_5_val_accuracies = []
        cnn_benchmark = []
        # for epoch,acc,loss,top_k_categorical_accuracy,val_acc,val_loss,val_top_k_categorical_accuracy in reader:
        for epoch, acc, loss, val_acc, val_loss in reader:
            accuracies.append(float(acc)*100)
            val_accuracies.append(float(val_acc)*100)
            # top_5_accuracies.append(float(top_k_categorical_accuracy)*100)
            # top_5_val_accuracies.append(float(val_top_k_categorical_accuracy)*100)
            # cnn_benchmark.append(65)

        # plt.plot(accuracies, label='acc')
        # # plt.plot(top_5_accuracies, label='top 5 acc')
        # # plt.plot(cnn_benchmark, label ='benchmark')
        #
        #
        # plt.plot(val_accuracies, label='val_acc')
        # # plt.plot(top_5_val_accuracies, label='top 5 val acc')
        # plt.plot(cnn_benchmark)
        # # plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # plt.xticks([0, 100, 200, 300, 400, 500, 600])
        # plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # plt.xlabel('Época')
        # plt.ylabel('Acurácia (%)')
        # plt.legend()
        # plt.show()

        print(max(val_accuracies, key=int))
if __name__ == '__main__':
    training_log = 'data/logs/logsMinhaBase/resnet152v2-3072-0.5.log'
    main(training_log)
