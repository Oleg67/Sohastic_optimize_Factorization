""" Multipurpose errors with niceties for printing and chaining """

class SimpleException(Exception):
    """ Exception to inherit for some more convenient argument handling """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args)
        self.__dict__.update(kwargs)
        if args:
            if isinstance(args[0], BaseException):
                original_error = args[0]
                self.original = original_error
                self.message = "%s: %s" % (type(original_error).__name__, original_error)
                if hasattr(original_error, 'job'):
                    self.job = original_error.job
            elif isinstance(args[0], (str, unicode)):
                try:
                    self.message = args[0] % args[1:]
                except TypeError:
                    self.message = args[0] + ': ' + str(args[1:])
            else:
                self.message = str(args[0]) if len(args) == 1 else str(args)

    def __str__(self):
        return self.message

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, ', '.join("%s=%s" % (attr, getattr(self, attr)) for attr in dir(self) if not attr.startswith('__')))

    @property
    def text(self):
        """ Keep this for some backwards compatibility """
        return self.message


class LogicError(SimpleException):
    """ Used when calculated values don't fit together """


class ValidationError(SimpleException):
    """ Used by properties to annonce invalid input """


class ValidationTypeError(ValidationError, TypeError):
    """ Special flavour so that a ValidationError can also be caught in generic except-clauses """


class ValidationValueError(ValidationError, ValueError):
    """ Special flavour so that a ValidationError can also be caught in generic except-clauses """


class TemporaryError(SimpleException):
    """
    All errors, that will probably fix themself after some waiting derive from this
    class. By checking for this error, calling functions can decide whether to retry 
    an operation or cancel and reraise an error. A timeout argument can be given in
    order to suggest a delay in seconds to the calling function.
    """
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop('timeout', 0)


class RetriesExceeded(SimpleException):
    """ Something with temporary errors got retried, but keeps on failing """


class ResetRequired(SimpleException):
    """ Restarting of some module or function is required """


class NotAvailable(SimpleException):
    """ Some external resource is not available """


class StatusError(SimpleException):
    """ Object is in the wrong state for this operation """


class ForbiddenStatusChange(StatusError):
    """ Transition from one state to another is not allowed """
    def __init__(self, obj, initial_state, target_state, comment=None, **kwargs):
        StatusError.__init__(self, **kwargs)
        self.message = '%s: %s -> %s' % (obj, initial_state, target_state)
        if comment:
            self.message += ': ' + comment


class SearchError(SimpleException):
    """ We tried to find something, but there were some issues """


class NotFound(SearchError):
    """ Something not there, even if it should be """


class MultipleEntries(SearchError):
    """ We found more than we expected """
    def __init__(self, entries, *args, **kwargs):
        SearchError.__init__(self, *args, **kwargs)
        self.entries = entries


class ConnectionError(SimpleException):
    """ All networking stuff goes here """


class TemporaryConnectionError(ConnectionError, TemporaryError):
    def __init__(self, *args, **kwargs):
        TemporaryError.__init__(self, *args, **kwargs)
        ConnectionError.__init__(self, *args, **kwargs)


class PermanentConnectionError(ConnectionError):
    """ ConnectionError which is not expected to fix itself in the near future """


class NetworkRetriesExceeded(PermanentConnectionError, RetriesExceeded):
    """ Too many temporary failures happened, giving up """


class EmptyResponse(TemporaryConnectionError):
    """ The connection was established, but no data returned """


class ParsingError(SimpleException, ValueError):
    """ Some major chunk of data could not be processed as expected """
