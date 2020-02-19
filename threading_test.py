from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import queue
import time
import threading
import os
import re
import sys
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')

SAVE_PATH = 'C:/TWCC/data/kid_data_Au/ptrain'
DATA_PATH = 'C:/TWCC/data/kid_data_Au/train'

SIZE = 299
D_I = []
D_O = []


def wav2png(IN_PATH, OUT_PATH):
    y, _ = librosa.load(IN_PATH)
    plt.figure()
    plt.figure(figsize=(1, 1), dpi=SIZE)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(OUT_PATH, bbox_inches='tight',
                pad_inches=0)
    plt.close()
    print('Save', OUT_PATH)


def save_data_to_array():
    i = 1
    for lable in os.listdir(DATA_PATH):
        for wavfile in os.listdir(DATA_PATH+'/'+lable):
            if("wav" in wavfile):
                i = i+1
                D_I.append(DATA_PATH+'/'+lable+'/'+wavfile)
                D_O.append(SAVE_PATH+'/'+lable+'/' +
                           wavfile.rsplit('.', -1)[0] + '.png')
    return i


class Worker(threading.Thread):
    def __init__(self, queue, num, semaphore):
        threading.Thread.__init__(self)
        self.queue = queue
        self.num = num
        self.semaphore = semaphore

    def run(self):
        while self.queue.qsize() > 0:
            msg = self.queue.get()

            # 取得旗標
            semaphore.acquire()
            # print("Semaphore acquired by Worker %d" % self.num)

            # 僅允許有限個執行緒同時進的工作
            print("Worker %d: %s" % (self.num, msg))
            print('job ', self.num, ' transform ',
                  D_I[self.num], ' to ', D_O[self.num])
            wav2png(D_I[self.num], D_O[self.num])

            # 釋放旗標
            # print("Semaphore released by Worker %d" % self.num)
            self.semaphore.release()


my_queue = queue.Queue()

thread_number = save_data_to_array()

for i in range(thread_number):
    my_queue.put("Job %d" % i)

# 建立旗標
semaphore = threading.Semaphore(8)


my_worker1 = Worker(my_queue, 1, semaphore)
my_worker2 = Worker(my_queue, 2, semaphore)
my_worker3 = Worker(my_queue, 3, semaphore)
my_worker4 = Worker(my_queue, 4, semaphore)
my_worker5 = Worker(my_queue, 5, semaphore)
my_worker6 = Worker(my_queue, 6, semaphore)
my_worker7 = Worker(my_queue, 7, semaphore)
my_worker8 = Worker(my_queue, 8, semaphore)

my_worker1.start()
my_worker2.start()
my_worker3.start()
my_worker4.start()
my_worker5.start()
my_worker6.start()
my_worker7.start()
my_worker8.start()

my_worker1.join()
my_worker2.join()
my_worker3.join()
my_worker4.join()
my_worker5.join()
my_worker6.join()
my_worker7.join()
my_worker8.join()

print("Done.")
