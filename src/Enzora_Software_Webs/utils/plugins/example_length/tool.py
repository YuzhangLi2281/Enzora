from Enzora_Agent.tools import AgentTool, ToolContext, ToolSpec


class SequenceLengthTool(AgentTool):
    def __init__(self, manifest):
        self.spec = ToolSpec(
            name=manifest["name"],
            version=manifest["version"],
            description=manifest["description"],
            capabilities=tuple(manifest["capabilities"]),
            inputs=tuple(manifest["inputs"]),
            outputs=tuple(manifest["outputs"]),
        )

    def run(self, context: ToolContext):
        return {"sequence_length": len(context.artifacts["protein_sequence"])}


def create_tool(manifest):
    return SequenceLengthTool(manifest)
