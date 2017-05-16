import time
import os
import cPickle as pickle
from copy import deepcopy
from collections import OrderedDict, MutableMapping


class AttrDict(dict):
    """ Syntax candy """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        """ Copy instance, preserving type """
        # Unfortunately, it's about a magnitude slower than dict.copy()
        return type(self)(self)

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(dict(self), memo))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, d):
        self.update(d)


class NoneTraversal(object):
    """ Object for pure laziness and syntax candy, allowing fake directory traversals, always just in returning a fake None in the end """

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__

    def __eq__(self, other):
        return other == None

    def __ne__(self, other):
        return other != None

    def __getattr__(self, attr):
        return self

    def __getitem__(self, key):
        return self

    def __str__(self):
        return str(None)


class DefaultNoneAttrDict(AttrDict):
    def __getattr__(self, attr):
        return self.get(attr, NoneTraversal())

    def __getitem__(self, key):
        return self.get(key, NoneTraversal())


class Folder(object):
    """ Wraps os and os.path calls into treelike accessible objects """

    def __init__(self, basedir='/', ignore_hidden=True):
        if not isinstance(basedir, str):
            basedir = str(basedir)
        if not os.path.exists(basedir) or not os.path.isdir(basedir):
            basedir = os.path.dirname(basedir)
        if not os.path.isdir(basedir):
            raise ValueError("Can not determine base path: %s" % basedir)
        self._basedir = basedir
        self._ignore_hidden = ignore_hidden

    def __getitem__(self, item):
        if item.startswith('.') and item not in ('.', '..') and self._ignore_hidden:
            raise ValueError('Hidden files are ignored')
        item_path = self.join(item)
        if not os.path.exists(item_path):
            item_lwr = item.lower()
            for fname in self.listdir():
                # We check if we have a match ignoring case and ending
                if fname.rsplit('.', 1)[0].lower() == item_lwr:
                    item_path = os.path.join(self._basedir, fname)
                    break
            else:
                raise KeyError(item_path)
        if os.path.isdir(item_path):
            return type(self)(item_path, ignore_hidden=self._ignore_hidden)
        elif os.path.isfile(item_path):
            return item_path
        else:
            raise TypeError("Found object is neither directory nor file: %s" % item_path)

    def __getattr__(self, item):
        """ Convenience function for easier item access """
        if item == '_basedir' or item.startswith('__'):
            raise AttributeError(item)
        return self.__getitem__(item)

    def __contains__(self, item):
        try:
            self.__getitem__(item)
        except KeyError:
            return False
        return True

    def __len__(self):
        return len(self.listdir())

    def __str__(self):
        return getattr(self, '_basedir', '')

    def __repr__(self):
        return "<Folder %s>" % getattr(self, '_basedir', '-')

    def join(self, *path):
        return os.path.abspath(os.path.join(self._basedir, *path))

    def listdir(self):
        return os.listdir(self._basedir)

    def subfolders(self):
        return AttrDict((name, type(self)(self.join(name))) for name in self.listdir()
                        if os.path.isdir(self.join(name)) and not (self._ignore_hidden and name.startswith('.')))

    def getfolder(self, item):
        """ Get or create subfolder """
        try:
            folder = self[item]
        except KeyError:
            os.mkdir(self.join(item))
            return self[item]
        else:
            if not isinstance(folder, type(self)):
                # Plain str returned for files
                raise KeyError("Folder name already taken")
            else:
                return folder

    def up(self):
        return self['..']
