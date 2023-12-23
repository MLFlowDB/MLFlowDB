import {Edge, MarkerType, Node} from "reactflow"

/// 边的 ID，由起点 ID 和终点 ID 组成
export const EdgeID = (source: string, target: string) => `${source}->${target}`

/// 节点属性的 ID
/// 对于数据节点，ID 由数据本身的 ID 和属性名组成
/// 对于操作节点，ID 由操作节点的 ID 和映射后的属性名组成
export const AttrID = (id: string, name: string) => `${id}_${name}`

///PROV-DM中节点的类型
export type PROV_DM = 'entity' | 'agent' | 'activity'

// /// 数据节点包所属的ML阶段
// export type DataCategory = 'raw-data' | 'data-collection' | 'structured-data' | 'model'
// /// 操作节点包含的类别
// export type OperCategory = 'data-process' | 'ml-activity'


export type ML_Phase = 'data-gathering' | 'data-transforming' |'dataset' | 'ml-activity' | 'ml-archving'

export type PROV_RELATION =  'wasGeneratedBy' | 'used' | 'wasDerivedFrom' | 'wasAttributeTo' | 'wasQuoted' | 'hadMember' | 'wasInformed'

export type EdgeCategory = 'node' | 'attr'

//节点的背景颜色
export const backgroundColor: Record<ML_Phase, string> = {
    "data-gathering": "#dae3f3",
    "data-transforming": "#fbe5d6",
    "ml-activity": "#f2f2f2",
    "dataset": "#fff2cc",
    "ml-archving": "#d6dce5",
}

/// 所有种类节点都有的基础数据
export type NodeData = {
    name: string,
    info?: any,
    phase:ML_Phase
    mldr_type:string
    type:PROV_DM,
    version:number
}

/// 数据节点特有的数据
export type EntityNodeData = NodeData & {
    stored_data?: Record<string, any>
}

/// 操作节点特有的数据
export type ActivityNodeData = NodeData & {
    mapping_data?: Record<string, string>,
}

///
export type AgentNodeData = NodeData & {
    agent_data?: Record<string, any>
}

/// 由后端传入的原始节点数据
export type RawNode = {
    id: string,
    mldr_type:string,
    version:number
} & (
        ({ type: 'entity' } & EntityNodeData) |
        ({ type: 'activity' } & ActivityNodeData) |
        ({ type: 'agent' } & AgentNodeData)
    )
/// 由后端传入的原始边数据
export type RawEdge = {
    type:PROV_RELATION,
    started :string,
    ended : string
}


/// React flow 的节点类型
export type WorkflowNode = Node<EntityNodeData | ActivityNodeData | AgentNodeData>

/// React flow 的边类型
export type WorkflowEdge = Edge<{category?:EdgeCategory}>

/// 根据原始边获取相应类型的节点数据
const processRawEdge = (rawEdge: RawEdge): WorkflowEdge => {
    const { type,started,ended} = rawEdge
    return {
        id: EdgeID(started, ended),
        source: started,
        sourceHandle: started,
        target: ended,
        targetHandle: ended,
        // type:type,
        hidden: false,
        label : type,
        data : { category:'node'},
        markerEnd: {
            type: MarkerType.ArrowClosed,
        },
        // animated: true
    }
}


/// 根据原始节点数据获取相应类型的节点数据
const getData = (rawNode: RawNode) => {
    const {  name,type, info,phase,mldr_type,version} = rawNode
    switch (type) {
        case 'entity':
            const { stored_data } = rawNode
            return { name,info, phase, stored_data,mldr_type,type,version} as EntityNodeData
        case 'activity':
            const { mapping_data } = rawNode
            return  {name,info, phase, mapping_data,mldr_type,type,version } as ActivityNodeData
        case 'agent':
            const { agent_data } = rawNode
            return { name,info, phase,agent_data,mldr_type,type,version} as AgentNodeData
    }
}

const processRawNode = (rawNode: RawNode): WorkflowNode => {
    const { id, type } = rawNode
    return {
        id,
        type,
        position: { x: 0, y: 0 },
        draggable: true,
        data: getData(rawNode)
    }
}

export default (rawNodes: RawNode[],rawEdges:RawEdge[]): { nodes: WorkflowNode[], edges: WorkflowEdge[] } => {
    const nodes: WorkflowNode[] = []
    // const lookup = new Map(rawNodes.map(node => [node.id, node]))
    const edges: WorkflowEdge[] = []

    console.log(rawEdges)

    for (const rawNode of rawNodes) {
        nodes.push(processRawNode(rawNode))
    }
    for (const rawEdge of rawEdges) {
        edges.push(processRawEdge(rawEdge))
    }
    return { nodes, edges }
}



export const rawNodes: RawNode[] = [
    {
        id: '1',
        name: '数据集',
        phase: 'dataset',
        mldr_type: 'Dataset',
        type: 'entity',
        stored_data: {},
        info : {'created':'this'},
        version : 1
    },
    {
        id: '2',
        name: '数据预处理',
        phase: 'data-transforming',
        mldr_type: 'Dataset',
        type: 'activity',
        mapping_data: {},
        version : 1
    },
    {
        id: '3',
        name: '代理',
        phase: 'data-transforming',
        mldr_type: 'Agent',
        type: 'agent',
        agent_data: {},
        version : 1
    }
]

export const rawEdges: RawEdge[] = [
    {
        type: 'wasAttributeTo',
        started:"2",
        ended:"3"
    },
    {
        type: 'wasGeneratedBy',
        started:"1",
        ended:"2"
    }

]