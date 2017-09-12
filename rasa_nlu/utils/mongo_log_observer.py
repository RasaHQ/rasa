from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

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
from twisted.internet import ssl

logger = logging.getLogger(__name__)


@implementer(ILogObserver)
class MongoLogObserver(object):
    """
    Log observer that writes to a Mongo DB.
    """

    def __init__(self, mongo_uri, collection_name, tls_ctx, format_event):
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

        self.formatEvent = format_event
        self.mongo_uri = mongo_uri
        self.dbname = self.get_db_name(mongo_uri)
        self.logs_collection = None
        self.collection_name = collection_name
        self.db = None
        if tls_ctx is None and 'ssl=true' in mongo_uri:
            # this will only work with username:password authentication.
            self.tls_ctx = ssl.ClientContextFactory()
        else:
            self.tls_ctx = tls_ctx

    @staticmethod
    def get_db_name(mongo_uri):
        regex = r"mongodb:\/\/(?P<hosts>[^\/]+)\/(?P<dbname>[^\?]+)\?(?P<options>.+)"
        regex = re.compile(regex)
        match = regex.search(mongo_uri)
        return match.groupdict()['dbname']

    @inlineCallbacks
    def connect(self):
        connection = yield ConnectionPool(self.mongo_uri, ssl_context_factory=self.tls_ctx)
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


def mongoLogObserver(mongo_uri, collection_name, tls_ctx=None):
    """
    Create a L{MongoLogObserver} that emits text to a specified (writable)
    Mongo DB.

    @param mongo_uri: A Mongo Uri
    @type mongo_uri: L{unicode}

    @param collection_name: A collection name
    @type collection_name: L{unicode}

    @param tls_ctx: An SSL context. See http://txmongo.readthedocs.io/en/latest/index.html
    @type tls_ctx: L{object}

    @return: A Mongo log observer.
    @rtype: L{MongoLogObserver}
    """
    observer = MongoLogObserver(
        mongo_uri,
        collection_name,
        tls_ctx,
        lambda event: eventAsJSON(event)
    )

    # We don't wanna do anything in the callback. But it's necessary to have one or async operations won't be triggered
    observer.connect().addCallback(lambda ign: ign).addErrback(logger.error)
    return observer

