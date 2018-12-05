from __future__ import division, print_function, absolute_import

import time
import sys

# Verify curses module for Windows and Notebooks Support
try:
    from IPython.core.display import clear_output
except:
    pass

CURSES_SUPPORTED = True
try:
    import curses
except Exception:
    print("curses is not supported on this machine (please install/reinstall curses for an optimal experience)")
    CURSES_SUPPORTED = False


class Callback(object):
    """ Callback base class. """
    def __init__(self):
        pass

    def on_train_begin(self, training_state):
        pass

    def on_epoch_begin(self, training_state):
        pass

    def on_batch_begin(self, training_state):
        pass

    def on_sub_batch_begin(self, training_state):
        pass

    def on_sub_batch_end(self, training_state, train_index=0):
        pass

    def on_batch_end(self, training_state, snapshot=False):
        pass

    def on_epoch_end(self, training_state):
        pass

    def on_train_end(self, training_state):
        pass


class ChainCallback(Callback):
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks

    def on_train_begin(self, training_state):
        for callback in self.callbacks:
            callback.on_train_begin(training_state)

    def on_epoch_begin(self, training_state):
        for callback in self.callbacks:
            callback.on_epoch_begin(training_state)

    def on_batch_begin(self, training_state):
        for callback in self.callbacks:
            callback.on_batch_begin(training_state)

    def on_sub_batch_begin(self, training_state):
        for callback in self.callbacks:
            callback.on_sub_batch_begin(training_state)

    def on_sub_batch_end(self, training_state, train_index=0):
        for callback in self.callbacks:
            callback.on_sub_batch_end(training_state, train_index)

    def on_batch_end(self, training_state, snapshot=False):
        for callback in self.callbacks:
            callback.on_batch_end(training_state, snapshot)

    def on_epoch_end(self, training_state):
        for callback in self.callbacks:
            callback.on_epoch_end(training_state)

    def on_train_end(self, training_state):
        for callback in self.callbacks:
            callback.on_train_end(training_state)

    def add(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")

        self.callbacks.append(callback)


class TermLogger(Callback):
    def __init__(self):
        self.data = []
        self.has_ipython = True
        self.display_type = "multi"
        self.global_data_size = 0
        self.global_val_data_size = 0
        self.snapped = False

        global CURSES_SUPPORTED
        if CURSES_SUPPORTED:
            try:
                curses.setupterm()
                sys.stdout.write(curses.tigetstr('civis').decode())
            except Exception:
                CURSES_SUPPORTED = False
        
        try:
            clear_output
        except NameError:
            self.has_ipython = False

    def add(self, data_size, val_size=0, metric_name=None, name=None):
        if not metric_name: metric_name = 'acc'
        self.data.append({
            'name': name if name else "Train op. " + str(len(self.data)),
            'metric_name': metric_name,
            'data_size': data_size,
            'epoch': 0,
            'step': 0,
            'val_size': val_size,
            'loss': None,
            'acc': None,
            'val_loss': None,
            'val_acc': None
        })
        self.global_data_size += data_size
        self.global_val_data_size += val_size

    def on_epoch_begin(self, training_state):
        training_state.step_time = time.time()
        training_state.step_time_total = 0.

    def on_epoch_end(self, training_state):
        pass

    def on_batch_begin(self, training_state):
        training_state.step_time = time.time()

    def on_batch_end(self, training_state, snapshot=False):

        training_state.step_time_total += time.time() - training_state.step_time
        if snapshot:
            self.snapshot_termlogs(training_state)
        else:
            self.print_termlogs(training_state)

    def on_sub_batch_begin(self, training_state):
        pass

    def on_sub_batch_end(self, training_state, train_index=0):

        self.data[train_index]['loss'] = training_state.loss_value
        self.data[train_index]['acc'] = training_state.acc_value
        self.data[train_index]['val_loss'] = training_state.val_loss
        self.data[train_index]['val_acc'] = training_state.val_acc
        self.data[train_index]['epoch'] = training_state.epoch
        self.data[train_index]['step'] = training_state.current_iter

    def on_train_begin(self, training_state):
        print("---------------------------------")
        print("Training samples: " + str(self.global_data_size))
        print("Validation samples: " + str(self.global_val_data_size))
        print("--")
        if len(self.data) == 1:
            self.display_type = "single"

    def on_train_end(self, training_state):
        # Reset caret to last position
        to_be_printed = ""
        if CURSES_SUPPORTED: #if not self.has_ipython #TODO:check bug here
            for i in range(len(self.data) + 2):
                to_be_printed += "\033[B"
            if not self.snapped:
                to_be_printed += "--\n"
        sys.stdout.write(to_be_printed)
        sys.stdout.flush()

        # Set caret visible if possible
        if CURSES_SUPPORTED:
            sys.stdout.write(curses.tigetstr('cvvis').decode())

    def termlogs(self, step=0, global_loss=None, global_acc=None, step_time=None):

        termlogs = "Training Step: " + str(step) + " "
        if global_loss:
            termlogs += " | total loss: \033[1m\033[32m" + \
                        "%.5f" % global_loss + "\033[0m\033[0m"
        if global_acc and not self.display_type == "single":
            termlogs += " - avg acc: %.4f" % float(global_acc)
        if step_time:
            termlogs += " | time: %.3fs" % step_time
        termlogs += "\n"
        for i, data in enumerate(self.data):
            print_loss = ""
            print_acc = ""
            print_val_loss = ""
            print_val_acc = ""
            if data['loss'] is not None:
                print_loss = " | loss: " + "%.5f" % data['loss']
            if data['acc'] is not None:
                print_acc = " - " + data['metric_name'] + ": " + \
                            "%.4f" % data['acc']
            if data['val_loss'] is not None:
                print_val_loss = " | val_loss: " + "%.5f" % data['val_loss']
            if data['val_acc'] is not None:
                print_val_acc = " - val_" + data['metric_name'] + ": " + "%.4f" % data['val_acc']
            # fix diplay, if step reached the whole epoch, display epoch - 1, as epoch has been updated
            print_epoch = data['epoch']
            # Smoothing display, so we show display at step + 1 to show data_size/data_size at end
            print_step = " -- iter: " + \
                         ("%0" + str(len(str(data['data_size']))) +
                          "d") % data['step'] + "/" + str(data['data_size'])
            if data['step'] == 0:
                print_epoch = data['epoch']
                # print_step = ""
                print_step = " -- iter: " + ("%0" + str(
                    len(str(data['data_size']))) + "d") % 0 \
                             + "/" + str(data['data_size'])
            termlogs += "\x1b[2K\r| " + data['name'] + " | epoch: " + \
                        "%03d" % print_epoch + print_loss + print_acc + \
                        print_val_loss + print_val_acc + print_step + "\n"

        return termlogs

    def print_termlogs(self, training_state):

        termlogs = self.termlogs(
            step=training_state.step,
            global_loss=training_state.global_loss,
            global_acc=training_state.global_acc,
            step_time=training_state.step_time_total)

        if self.has_ipython and not CURSES_SUPPORTED:
            clear_output(wait=True)
        else:
            for i in range(len(self.data) + 1):
                termlogs += "\033[A"

        sys.stdout.write(termlogs)
        sys.stdout.flush()

    def snapshot_termlogs(self, training_state):

        termlogs = self.termlogs(
            step=training_state.step,
            global_loss=training_state.global_loss,
            global_acc=training_state.global_acc,
            step_time=training_state.step_time_total)

        termlogs += "--\n"

        sys.stdout.write(termlogs)
        sys.stdout.flush()
        self.snapped = True


class ModelSaver(Callback):
    def __init__(self, save_func, snapshot_path, best_snapshot_path,
                 best_val_accuracy, snapshot_step, snapshot_epoch):
        self.save_func = save_func
        self.snapshot_path = snapshot_path
        self.snapshot_epoch = snapshot_epoch
        self.best_snapshot_path = best_snapshot_path
        self.best_val_accuracy = best_val_accuracy
        self.snapshot_step = snapshot_step

    def on_epoch_begin(self, training_state):
        pass

    def on_epoch_end(self, training_state):
        if self.snapshot_epoch:
            self.save(training_state.step)

    def on_batch_begin(self, training_state):
        pass

    def on_batch_end(self, training_state, snapshot=False):

        if snapshot & (self.snapshot_step is not None):
            self.save(training_state.step)

        if None not in (self.best_snapshot_path, self.best_val_accuracy, training_state.val_acc):
            if training_state.val_acc > self.best_val_accuracy:
                self.best_val_accuracy = training_state.val_acc
                self.save_best(int(10000 * round(training_state.val_acc, 4)))

    def on_sub_batch_begin(self, training_state):
        pass

    def on_sub_batch_end(self, training_state, train_index=0):
        pass

    def on_train_begin(self, training_state):
        pass

    def on_train_end(self, training_state):
        pass

    def save(self, training_step=0):
        if self.snapshot_path:
            self.save_func(self.snapshot_path, training_step)

    def save_best(self, val_accuracy):
        if self.best_snapshot_path:
            snapshot_path = self.best_snapshot_path + str(val_accuracy)
            self.save_func(snapshot_path, use_val_saver=True)
