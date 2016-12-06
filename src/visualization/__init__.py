from itertools import cycle, groupby


def entity_spec(name, red, green, blue):
    return """
[data-entity][data-entity="{name}"] {{
    background: rgba({r}, {g}, {b}, 0.2);
    border-color: rgb({r}, {g}, {b});
}}

[data-entity][data-entity="{name}"]::after {{
    background: rgb({r}, {g}, {b});
}}""".format(name=name, r=red, g=green, b=blue)


def create_css(entity_examples):
    colors = cycle([(166, 226, 45), (67, 198, 252), (47, 187, 171)])
    entity_types = set()
    for example in entity_examples:
        for entity in example["entities"]:
            entity_types.add(entity["entity"])
    entity_specs = u"\n".join([entity_spec(name, *next(colors)) for name in entity_types])
    return u"""
    <style media="screen" type="text/css">
    .entities {{
        line-height: 2;
    }}
    [data-entity] {{
        padding: 0.25em 0.35em;
         margin: 0px 0.25em;
         line-height: 1;
         display: inline-block;
         border-radius: 0.25em;
         border: 1px solid;
    }}

    [data-entity]::after {{
        box-sizing: border-box;
        content: attr(data-entity);
        font-size: 0.6em;
        line-height: 1;
        padding: 0.35em;
        border-radius: 0.35em;
        text-transform: uppercase;
        display: inline-block;
        vertical-align: middle;
        margin: 0px 0px 0.1rem 0.5rem;
    }}
    {entity_specs}
    </style>""".format(entity_specs=entity_specs)


def html_wrapper():
    return u"""
<!DOCTYPE html>
<html>
  <head>
    {head}
  </head>
  <body>
    {body}
  </body>
</html>
    """


def format_example(example):
    text = example["text"]
    entities = sorted(example["entities"], key=lambda e: e['start'])
    chunks = []
    end = 0
    while len(entities) > 0:
        entity = entities.pop(0)
        start = entity["start"]
        chunks.append(text[end:start])
        end = entity["end"]
        markup = u"""<mark data-entity="{0}">{1}</mark>""".format(entity["entity"], text[start:end])
        chunks.append(markup)
    chunks.append(text[end:])
    return u"<div>{0}</div>".format(u"".join(chunks))

def intent_group(examples):
    content = u"<h4>{0}</h4>".format(examples[0]["intent"])
    content += u"".join([format_example(example) for example in examples])
    return content

def create_html(training_data):
    #list(set(training_data.entity_examples + training_data.intent_examples))
    examples = sorted(training_data.entity_examples, key=lambda e:e["intent"])
    intentgroups = []
    for _, group in groupby(examples, lambda e: e["intent"]):
        intentgroups.append(list(group))

    body = u"".join([intent_group(g) for g in intentgroups])
    head = create_css(examples)
    template = html_wrapper()
    return template.format(head=head, body=body)
