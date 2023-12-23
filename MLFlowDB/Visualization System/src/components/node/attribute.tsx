import { Handle, Position } from "reactflow"

type AttributeProps = {
    name: string
    value?: string
    sourceID: string
    targetID: string
}
export default ({ name, value, sourceID, targetID }: AttributeProps) => {
    return <div className="relative">
        <Handle type="target" position={Position.Left} id={sourceID} />
        <div className="
            w-full h-[50px]
            bg-white rounded-lg shadow-lg
            flex items-center justify-center
            text-sm overflow-hidden">
            {name}{value ? ` : ${value}` : ''}
        </div>
        <Handle type="source" position={Position.Right} id={targetID} />
    </div>
}