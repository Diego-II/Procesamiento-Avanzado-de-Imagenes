import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys
import torch

def train_for_classification(net, train_loader, test_loader, optimizer, 
                             criterion, lr_scheduler=None,epochs=1, reports_every=1, device='cuda'):
  net.to(device)
  total_train = len(train_loader.dataset)
  total_test = len(test_loader.dataset)
  tiempo_epochs = 0
  train_loss, train_acc, test_acc = [], [], []

  for e in range(1,epochs+1):  
    inicio_epoch = timer()
    
    # Aseguramos que todos los parámetros se entrenarán usando .train()
    net.train()

    # Variables para las métricas
    running_loss, running_acc = 0.0, 0.0

    for i, data in enumerate(train_loader):
      # Desagregamos los datos y los pasamos a la GPU
      X, Y = data
      X, Y = X.to(device), Y.to(device)

      # Limpiamos los gradientes, pasamos el input por la red, calculamos
      # la loss, ejecutamos el backpropagation (.backward) 
      # y un paso del optimizador para modificar los parámetros
      optimizer.zero_grad()

      y_hat = net(X)
      loss = criterion(y_hat, Y)

      
      loss.backward()
      optimizer.step()

      # loss
      items = min(total_train, (i+1) * train_loader.batch_size)
      running_loss += loss.item()
      avg_loss = running_loss/(i+1)
      
      # accuracy
      _, max_idx = torch.max(y_hat, dim=1)
      running_acc += torch.sum(max_idx == Y).item()
      avg_acc = running_acc/items*100

      # report
      sys.stdout.write(f'\rEpoch:{e}({items}/{total_train}), ' 
                       + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                       + f'Loss:{avg_loss:02.5f}, '
                       + f'Train Acc:{avg_acc:02.1f}%')
      
    tiempo_epochs += timer() - inicio_epoch

    if e % reports_every == 0:
      sys.stdout.write(', Validating...')
      train_loss.append(avg_loss)
      train_acc.append(avg_acc)
      net.eval()
      running_acc = 0.0
      for i, data in enumerate(test_loader):
        X, Y = data
        X, Y = X.to(device), Y.to(device)
        y_hat = net(X)
        _, max_idx = torch.max(y_hat, dim=1)
        running_acc += torch.sum(max_idx == Y).item()
        avg_acc = running_acc/total_test*100
      test_acc.append(avg_acc)
      sys.stdout.write(f', Val Acc:{avg_acc:02.2f}%, '
                       + f'Avg-Time:{tiempo_epochs/e:.3f}s.\n')
    else:
      sys.stdout.write('\n')

    if lr_scheduler is not None:
      lr_scheduler.step()

  return train_loss, (train_acc, test_acc)

def plot_results(loss, score1, score1_title='Accuracy', score2=None, score2_title=None):
  f1 = plt.figure(1)
  ax1 = f1.add_subplot(111)
  ax1.set_title("Loss")    
  ax1.set_xlabel('epochs')
  ax1.set_ylabel('loss')
  ax1.plot(loss, c='r')
  ax1.legend(['train-loss'])
  f1.show()

  f2 = plt.figure(2)
  ax2 = f2.add_subplot(111)
  ax2.set_title(score1_title)    
  ax2.set_xlabel('epochs')
  ax2.set_ylabel(score1_title.lower())
  ax2.plot(score1[0], c='b')
  ax2.plot(score1[1], c='g')
  ax2.legend([f'train-{score1_title.lower()}', f'val-{score1_title.lower()}'])
  f2.show()

  if score2:
    f3= plt.figure(3)
    ax3 = f3.add_subplot(111)
    ax3.set_title(score2_title)    
    ax3.set_xlabel('epochs')
    ax3.set_ylabel(score2_title.lower())
    ax3.plot(score2[0], c='b')
    ax3.plot(score2[1], c='g')
    ax3.legend([f'train-{score2_title.lower()}', f'val-{score2_title.lower()}'])
    f3.show()

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
