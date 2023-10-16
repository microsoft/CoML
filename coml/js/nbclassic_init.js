function getDebugInformation() {
    const runningCells = $(".running");
    if (runningCells.length === 0) {
        console.warn("No running cell");
        return null;
    }
    const cellIndex = Jupyter.notebook.get_cell_elements().index(runningCells[0]);
    if (cellIndex <= 0) {
        console.warn("No previous cell");
        return null;
    }
    const cell = IPython.notebook.get_cell(cellIndex - 1);
    const cellDump = cell.toJSON();
    return cellDump;
}

IPython.CodeCell.prototype.native_handle_input_request = IPython.CodeCell.prototype.native_handle_input_request || IPython.CodeCell.prototype._handle_input_request;
IPython.CodeCell.prototype._handle_input_request = function (msg) {
    try {
        // only apply the hack if the command is valid JSON
        const command = JSON.parse(msg.content.prompt);
        const kernel = IPython.notebook.kernel;
        if (command["command"] === "last_cell") {
            kernel.send_input_reply(JSON.stringify(getDebugInformation()));
        } else {
            console.log("Not a command", msg);
            this.native_handle_input_request(msg);
        }
    } catch(err) {
        console.log("Not a command", msg, err);
        this.native_handle_input_request(msg);
    }
}