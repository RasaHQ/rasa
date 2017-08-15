from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""
Mongo log observer.
"""
import logging

from twisted.internet.defer import inlineCallbacks
from zope.interface import implementer

from twisted.python.compat import StringType
from txmongo.connection import ConnectionPool
from txmongo.collection import Collection
from twisted.logger._file import ILogObserver
from twisted.logger._json import eventAsJSON

logger = logging.getLogger(__name__)

@implementer(ILogObserver)
class MongoLogObserver(object):
    """
    Log observer that writes to a Mongo DB.
    """

    def __init__(self, mongo_uri, tls_ctx, format_event):
        """
        @param mongo_uri: A Mongo Uri
        @type mongo_uri: L{unicode}

        @param tls_ctx: An SSL context. See http://txmongo.readthedocs.io/en/latest/index.html
        @type tls_ctx: L{object}

        @param format_event: A callable that formats an event.
        @type format_event: L{callable} that takes an C{event} argument and
            returns a formatted event as L{unicode}.
        """
        assert isinstance(mongo_uri, StringType)
        # assert len(mongo_uri.rsplit('/', 1)) > 1 # make sure db name is appended to uri

        self._encoding = "utf-8"
        self.tls_ctx = tls_ctx
        self.formatEvent = format_event
        self.mongoUri = mongo_uri.rsplit('/', 1)[0]
        self.dbname = mongo_uri.rsplit('/', 1)[1]
        self.logs_collection = None
        self.collection_name = u'rasa_nlu_logs'
        self.db = None

    @inlineCallbacks
    def connect(self):
        connection = yield ConnectionPool(self.mongoUri)
        self.db = getattr(connection, self.dbname)
        collection_names = yield self.db.command("listCollections")
        values = collection_names['cursor']['firstBatch']
        if len(filter(lambda i: i['type'] == u'collection' and i['name'] == self.collection_name, values)) == 0:
            self.logs_collection = yield self.db.command("create", self.collection_name)

    def __call__(self, event):
        """
        Write event to Mongo db.

        @param event: An event.
        @type event: L{dict}
        """

        if event is None:
            return

        event['log_level'] = event['log_level'].name
        event['log_logger'] = event['log_logger'].namespace
        collection = Collection(self.db, self.collection_name)
        collection.insert_one(event)


def print_error(failure):
    print (str(failure))


def mongoLogObserver(mongo_uri, tls_ctx=None):
    """
    Create a L{MongoLogObserver} that emits text to a specified (writable)
    Mongo DB.

    @param mongo_uri: A Mongo Uri
    @type mongo_uri: L{unicode}

    @param tls_ctx: An SSL context. See http://txmongo.readthedocs.io/en/latest/index.html
    @type tls_ctx: L{object}

    @return: A file log observer.
    @rtype: L{FileLogObserver}
    """
    observer = MongoLogObserver(
        mongo_uri,
        tls_ctx,
        lambda event: eventAsJSON(event)
    )

    # We don't wanna do anything in the callback. But it's necessary to have one or async operations won't be triggered
    observer.connect().addCallback(lambda ign: ign).addErrback(logger.error)
    return observer

