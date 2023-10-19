window.comlGetRunningCellIndex = function () {
    const runningCells = $(".running");
    if (runningCells.length === 0) {
        console.warn("No running cell");
        return null;
    }
    const cellIndex = Jupyter.notebook.get_cell_elements().index(runningCells[0]);
    if (cellIndex < 0) {
        console.error("Running cell not found in cell list.");
        return null;
    }
    return cellIndex;
}

window.comlGetCurrentCell = function () {
    const cell = comlGetRunningCellIndex();
    if (cell === null) {
        return null;
    }
    return IPython.notebook.get_cell(cell);
}

window.comlGetLastCell = function () {
    const cellIndex = comlGetRunningCellIndex();
    if (cellIndex === null) {
        return null;
    }
    return IPython.notebook.get_cell(comlGetRunningCellIndex() - 1);
}

if (window.IPython && IPython.CodeCell) {
    window.IPythonAvailable = true;
    IPython.CodeCell.prototype.native_handle_input_request = IPython.CodeCell.prototype.native_handle_input_request || IPython.CodeCell.prototype._handle_input_request;
    IPython.CodeCell.prototype._handle_input_request = function (msg) {
        try {
            // only apply the hack if the command is valid JSON
            const command = JSON.parse(msg.content.prompt);
            const kernel = IPython.notebook.kernel;
            if (command["command"] === "last_cell") {
                kernel.send_input_reply(JSON.stringify(comlGetLastCell().toJSON()));
            } else if (command["command"] === "running_cell") {
                kernel.send_input_reply(JSON.stringify(comlGetCurrentCell().toJSON()));
            } else {
                console.log("Not a command", msg);
                this.native_handle_input_request(msg);
            }
        } catch(err) {
            console.log("Not a command", msg, err);
            this.native_handle_input_request(msg);
        }
    }
}
