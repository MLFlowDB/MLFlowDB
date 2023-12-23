import SideTable from "./components/side-table"
import Workflow from "./components/workflow"
import axios from "axios";
import {useEffect, useState} from "react";
import {rawEdges, rawNodes} from "@/utils/process";

const fetchData = async (uid: string) => {
    try {
        // 使用 Axios 发起本地接口请求
        const response = await axios.get('http://127.0.0.1:8888/api/trace/trace_result' + uid + '/');
        const result = response.data;

        // console.log(result)

        return result

    } catch (error) {
        console.error('Error fetching data:', error);
    }
};

export default () => {
    const pathName = window.location.pathname
    const [result, setResult] = useState({"nodes":[],"edges":[]});


    useEffect(() => {
        // 在组件挂载时获取数据
        fetchData(pathName)
            .then(data => {
                // console.log(data.data)
                setResult(data.data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }, []);

    return <div className="w-screen h-screen bg-gray-200 flex items-center justify-center">
        <div className="w-[100vw] h-[100vh] bg-white shadow-lg flex flex-row">
            {result.nodes.length !== 0 && (
                <>
                    <div className="w-3/4">
                        <Workflow datas={result}/>
                    </div>
                    <div className="w-1/4 h-full shadow-lg">
                        <SideTable/>
                    </div>
                </>)}
        </div>
    </div>
}