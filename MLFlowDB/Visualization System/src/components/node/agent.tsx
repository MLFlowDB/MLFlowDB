import {AttrID, AgentNodeData, backgroundColor} from "@/utils/process"
import { NodeProps } from "reactflow"
import Node from '.'
import Attribute from "./attribute"

export type AgentNodeProps = NodeProps<AgentNodeData>

export default (props: NodeProps<AgentNodeData>) => {
    const { id, data: { phase, agent_data, mldr_type,type } } = props
    return <Node<AgentNodeData> {...props} backgroundColor={backgroundColor[phase]} borderRadius={"50px"}>
        {Object.entries(agent_data ?? {})
            .map(([name, value]) =>
                <Attribute
                    key={AttrID(id, name)}
                    sourceID={AttrID(id, name)}
                    targetID={AttrID(id, name)}
                    name={name}
                    value={value} />)}
    </Node>
}