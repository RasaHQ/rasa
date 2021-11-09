from datadog import initialize, statsd
import time

options = {
    'statsd_host': '127.0.0.1',
    'statsd_port': 8125,
    'statsd_constant_tags': ['branch:invented'],
}

initialize(**options)

# while(1):
#     print('example_metric')
#     statsd.increment('example_metric.increment', tags=["environment:dev"])
#     statsd.decrement('example_metric.decrement', tags=["environment:dev"])
#     time.sleep(10)

for i in range(10):
  print('example_metric2')
  statsd.gauge('example_metric2.gauge', i, tags=["environment:dev"])
  time.sleep(10)
