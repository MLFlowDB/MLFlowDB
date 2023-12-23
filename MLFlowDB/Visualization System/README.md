# MLFlowDB Visualization

This directory includes a visualization tool for viewing the traceability graph of MLFlowDB. You can use this tool to visually inspect the tracking results of MLFlowDB for the ML lifecycle.

## Getting the code

You can download a copy of all the files in this repository by cloning the [git]([MLFlowDB/MLFlowDB (github.com)](https://github.com/MLFlowDB/MLFlowDB)) repository:

```
git clone https://github.com/MLFlowDB/MLFlowDB.git
```

## Dependencies

The system uses Yarn as the package management tool and Vite as the front-end toolchain.

Yarn depends on the Node.js environment. If you haven't installed Node.js yet, you can download and install it from the Node.js official website. Choose the version suitable for your operating system.

To install Yarn, open a command line tool and run the following command:

```
npm install -g yarn
```

To install packages, run the following code in the current directory：

```
yarn
```

## Re-running

After installation is complete, use Vite to start the project：

```
yarn run vite
```

## How to Use

After calling the trace interface of MLFlowDB, MLFlowDB will return the following content：

```json
{
  "status": "200",
  "data": {"query_id": <query_id>}
}
```

Our visualization system is by default deployed on port 5173. You can view the traceability result graph by visiting `http://localhost:5173/<query_id>`.

![image-20231221180148644](https://github.com/MLFlowDB/MLFlowDB/blob/main/MLFlowDB/Visualization%20System/README.assets/image-20231221180135106.png?raw=true)

After clicking on a node, detailed information about the node will be displayed.

![image-20231221180216825](https://github.com/MLFlowDB/MLFlowDB/blob/main/MLFlowDB/Visualization%20System/README.assets/image-20231221180216825.png?raw=true)