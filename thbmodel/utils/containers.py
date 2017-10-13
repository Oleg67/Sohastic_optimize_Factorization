import time
import os
import cPickle as pickle
from copy import deepcopy
from collections import OrderedDict, MutableMapping, namedtuple


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


class TwoWayDict(dict):
    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        dict.__setitem__(self, val, key)


class MultiDict(MutableMapping):
    __slots__ = 'data',

    def __init__(self, *args, **kwargs):
        self.data = dict()
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data.setdefault(key, []).append(val)

    def __delitem__(self, key):
        del self.data[key]


class AttrDictProxy(MutableMapping):
    """ Some more syntax candy for arbitrary mapping types """

    def __init__(self, source):
        object.__setattr__(self, '__source__', source)

    def __getitem__(self, key):
        return self.__source__.__getitem__(key)

    def __setitem__(self, key, val):
        self.__source__.__setitem__(key, val)

    def __delitem__(self, key):
        self.__source__.__delitem__(key)

    def __len__(self):
        return len(self.__source__)

    def __iter__(self):
        return iter(self.__source__)

    def __str__(self):
        return str(self.__source__)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __delattr__ = __delitem__


class OrderedAttrDict(OrderedDict):
    def __getattribute__(self, attr):
        try:
            return OrderedDict.__getattribute__(self, attr)
        except AttributeError:
            pass
        try:
            return self[attr]
        except KeyError:
            raise AttributeError("Attribute/key not found: %s" % attr)


class Storeable(object):
    def store(self, filename):
        with open(filename, 'wb') as outfile:
            try:
                pickle.dump(self, outfile, protocol=2)
            except pickle.PicklingError:
                import types
                import copy_reg
                import pdb

                def debug(obj):
                    pdb.set_trace()

                copy_reg.pickle(types.MethodType, debug)
                pickle.dumps(self, protocol=2)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as infile:
            return pickle.load(infile)


class Cache(dict, Storeable):
    """ For stuff that needs to be gone after a while """

    def __init__(self, length, max_age=None, intense_check_interval=30 * 60):
        dict.__init__(self)
        self.length = length
        self.max_age = max_age
        self.intense_check_interval = intense_check_interval
        self._last_clearing = 0

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        timestamp, val = dict.__getitem__(self, key)
        if self.max_age and timestamp < time.time() - self.max_age:
            del self[key]
            raise KeyError(key)
        self._housekeeping()
        return val

    def update_time(self, key):
        return dict.__getitem__(self, key)[0]

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, (time.time(), val))
        self._housekeeping()

    def _housekeeping(self, intense=None):
        if intense is None:
            # Regularly do intense cleanings, if intense is not explicitly specified
            intense = time.time() - self._last_clearing > self.intense_check_interval
        if len(self) > self.length:
            self._clear_outdated_entries()
            if len(self) > self.length:
                items = sorted(self.items(), key=lambda x: x[1][0])
                for cnt in range(len(self) - self.length):
                    del self[items[cnt][0]]
        elif intense:
            self._clear_outdated_entries()

    def _clear_outdated_entries(self):
        self._last_clearing = time.time()
        deadline = time.time() - self.max_age
        items = list(self.iteritems())
        for key, (timestamp, _) in items:
            if timestamp < deadline:
                try:
                    del self[key]
                except KeyError:
                    pass

    def update(self, *dicts, **kwargs):
        for d in dicts:
            for k, v in d.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v


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



class ParamsContainer(object):
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, val in kwargs.iteritems():
            setattr(self, key, val)
        return self

    def __setattr__(self, attr, val):
        attrs = [a for a in dir(type(self)) if not a.startswith('_')]
        valid_attrs = set(attrs).union(['_' + a for a in attrs])
        if attr in valid_attrs:
            # Verify, that a default parameter for a value actually exists to
            # avoid typing errors
            super(ParamsContainer, self).__setattr__(attr, val)
        else:
            raise AttributeError(attr)

    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                yield attr

    def _tuple_type(self):
        cls = type(self)
        try:
            # Make sure this is not retrieved from the super class
            return cls.__dict__['_tuple_type_cache']
        except KeyError:
            tuple_type = namedtuple(cls.__name__ + 'Tuple', sorted(self))
            cls._tuple_type_cache = tuple_type
            return tuple_type

    def namedtuple(self):
        """ Takes all the class variables of the instance <obj>, and puts them into a named tuple, ignoring the methods """
        new_dict = {}
        for attr in self:
            val = getattr(self, attr)
            if hasattr(val, 'name'):
                raise ValueError("Invalid attribute for jitting: %s" % attr)
            if isinstance(val, set):
                val = list(val)
            new_dict[attr] = val
        return self._tuple_type()(**new_dict)

    def __str__(self):
        out = type(self).__name__ + '\n'
        maxlen = max(len(attr) for attr in self) + 2
        for attr in self:
            values = str(getattr(self, attr))
            # get rid of annoying courses list in strategy printing
            if attr == 'event_courses_allowed' and getattr(self, attr) == range(89):
                values = 'all'
            out += '    ' + (attr + ': ').ljust(maxlen) + values + '\n'
        return out

    @classmethod
    def from_tuple(cls, ntuple, **kwargs):
        return cls(**ntuple._asdict()).update(**kwargs)


