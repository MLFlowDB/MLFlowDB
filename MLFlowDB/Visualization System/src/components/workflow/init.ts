import { Node, NodeProps, Edge } from 'reactflow'

export const nodes: Node[] = [
    {
        type: 'data',
        id: '1',
        data: { category: 'raw-data' },
        position: { x: 0, y: 0 },
        draggable: false,
    },
    {
        type: 'data',
        id: '2',
        data: { category: 'raw-data' },
        position: { x: 0, y: 0 },
    },
    {
        type: 'data',
        id: '3',
        data: { category: 'raw-data' },
        position: { x: 0, y: 0 },
    },
]

export const edges: Edge[] = [
    {
        id: '1->2',
        source: '1',
        target: '2',
    },
    {
        id: '1->3',
        source: '1',
        target: '3',
    },
]