from rasa.engine.recipes.recipe import Recipe


class GraphV1Recipe(Recipe):
    """Recipe which converts the graph model config to train and predict graph."""

    name = "graph.v1"
