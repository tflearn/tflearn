from __future__ import division, print_function, absolute_import

import sys, curses
try:
    from IPython.core.display import clear_output
except:
    pass


class Callback(object):
    """ Callback base class. """

    def __init__(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_sub_epoch_begin(self, **kwargs):
        pass

    def on_sub_epoch_end(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_sub_batch_begin(self, **kwargs):
        pass

    def on_sub_batch_end(self, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


class TermLogger(Callback):
    def __init__(self, training_step=0):
        super(TermLogger, self).__init__()
        self.data = []
        self.has_curses = True
        self.has_ipython = True
        self.display_type = "multi"
        self.global_loss = None
        self.global_acc = None
        self.global_step = training_step
        self.global_data_size = 0
        self.global_val_data_size = 0
        self.snapped = False
        try:
            curses.setupterm()
            sys.stdout.write(curses.tigetstr('civis').decode())
        except Exception:
            self.has_curses = False
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

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_sub_epoch_begin(self):
        pass

    def on_sub_epoch_end(self, snapshot=False):
        if snapshot:
            self.snapshot_termlogs()

    def on_batch_begin(self):
        pass

    def on_batch_end(self, global_loss=None, global_acc=None, snapshot=False):
        self.global_step += 1
        self.global_loss = global_loss
        self.global_acc = global_acc
        self.print_termlogs()
        if snapshot:
            self.snapshot_termlogs()

    def on_sub_batch_start(self):
        pass

    def on_sub_batch_end(self, train_op_i, epoch, step, loss=None, acc=None,
                         val_loss=None, val_acc=None):
        self.data[train_op_i]['loss'] = loss
        self.data[train_op_i]['acc'] = acc
        self.data[train_op_i]['val_loss'] = val_loss
        self.data[train_op_i]['val_acc'] = val_acc
        self.data[train_op_i]['epoch'] = epoch
        self.data[train_op_i]['step'] = step

    def on_train_begin(self):
        print("---------------------------------")
        print("Training samples: " + str(self.global_data_size))
        print("Validation samples: " + str(self.global_val_data_size))
        print("--")
        if len(self.data) == 1:
            self.display_type = "single"

    def on_train_end(self):
        # Reset caret to last position
        to_be_printed = ""
        if self.has_curses: #if not self.has_ipython #TODO:check bug here
            for i in range(len(self.data) + 2):
                to_be_printed += "\033[B"
            if not self.snapped:
                to_be_printed += "--\n"
        sys.stdout.write(to_be_printed)
        sys.stdout.flush()

        # Set caret visible
        if self.has_curses:
            sys.stdout.write(curses.tigetstr('cvvis').decode())

    def termlogs(self):

        termlogs = "Training Step: " + str(self.global_step) + " "
        if self.global_loss:
            termlogs += " | total loss: \033[1m\033[32m" + \
                        "%.5f" % self.global_loss + "\033[0m\033[0m"
        if self.global_acc and not self.display_type == "single":
            termlogs += " - avg acc: %.4f" % float(self.global_acc)
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
                print_val_acc = " - val_acc: " + "%.4f" % data['val_acc']
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

    def print_termlogs(self):

        termlogs = self.termlogs()
        if self.has_ipython and not self.has_curses:
            clear_output(wait=True)
        else:
            for i in range(len(self.data) + 1):
                termlogs += "\033[A"

        sys.stdout.write(termlogs)
        sys.stdout.flush()

    def snapshot_termlogs(self):

        termlogs = self.termlogs()
        termlogs += "--\n"

        sys.stdout.write(termlogs)
        sys.stdout.flush()
        self.snapped = True


class ModelSaver(object):
    def __init__(self, save_func, training_step, snapshot_path, snapshot_epoch):
        self.save_func = save_func
        self.training_step = training_step
        self.snapshot_path = snapshot_path
        self.snapshot_epoch = snapshot_epoch

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        if self.snapshot_epoch:
            self.save()

    def on_sub_epoch_begin(self):
        pass

    def on_sub_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self, snapshot_model=False):
        self.training_step += 1
        if snapshot_model:
            self.save()

    def on_sub_batch_begin(self):
        pass

    def on_sub_batch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def save(self):
        if self.snapshot_path:
            self.save_func(self.snapshot_path, self.training_step)
