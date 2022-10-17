---
sidebar_label: rasa.core.jobs
title: rasa.core.jobs
---
#### scheduler

```python
async scheduler() -> AsyncIOScheduler
```

Thread global scheduler to handle all recurring tasks.

If no scheduler exists yet, this will instantiate one.

#### kill\_scheduler

```python
kill_scheduler() -> None
```

Terminate the scheduler if started.

Another call to `scheduler` will create a new scheduler.

