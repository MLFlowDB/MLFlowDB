import { AttrID, backgroundColor, ActivityNodeData } from "@/utils/process"
import { NodeProps } from "reactflow"
import Node from '.'
import Attribute from "./attribute"

export type ActivityNodeProps = NodeProps<ActivityNodeData>

export default (props: NodeProps<ActivityNodeData>) => {
    const { id,  data: { phase, mapping_data, mldr_type,type} } = props
    return <Node<ActivityNodeData> {...props} backgroundColor={backgroundColor[phase]} objectTransform={"skewX(-30deg)"} contentTransform={"skewX(30deg)"}>
        {Object.entries(mapping_data ?? {})
            .map(([mapping, mapped]) =>
                <Attribute
                    key={`${id}_${mapped}`}
                    sourceID={AttrID(id, mapping)}
                    targetID={AttrID(id, mapped)}
                    name={mapping}
                    value={mapped} />)}
    </Node>
}