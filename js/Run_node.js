// ComfyUI.mxToolkit.Stop v.0.9.7 - Max Smirnov 2024
import { app } from "../../scripts/app.js";

class Run_node
{
    constructor(node)
    {
        this.node = node;
        /*
        this.node.properties = this.node.properties || {};

        this.node.onGraphConfigured = function ()
        {
            this.configured = true;
        }
        */
        this.node.onAdded = function ()
        {
            this.widgets[0].hidden = true;
            app.canvas.setDirty(true, true)
        };

        this.node.onMouseDown = function(e, pos, canvas)
        {
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            if ( e.canvasY-this.pos[1] > 0 ) return false;
            if (this.flags.collapsed && (e.canvasX-this.pos[0] < LiteGraph.NODE_TITLE_HEIGHT)) return false;
            if (!this.flags.collapsed && ((e.canvasX-this.pos[0]) < (this.size[0]-cWidth+LiteGraph.NODE_TITLE_HEIGHT))) return false;
            this.updateThisNodeGraph?.();
            this.onTmpMouseUp(e, pos, canvas);
            return true;
        }

        this.node.onTmpMouseUp = async function(e, pos, canvas)
        {
            const graph = window.graph;
            const nodes = graph._nodes;
            const originalModes = new Map();
            nodes.forEach(node => {
                originalModes.set(node.id, node.mode); // Lưu mode ban đầu theo ID của node
            });
            nodes.forEach(node => {
                if (node.selected) {
                    node.mode = LiteGraph.ALWAYS;
                }else{
                    node.mode = LiteGraph.NEVER;
                }
            });
            await app.queuePrompt(0);
            nodes.forEach(node => {
                const originalMode = originalModes.get(node.id);
                if (originalMode !== undefined) {
                    node.mode = originalMode; // Khôi phục mode từ trạng thái đã lưu
                }
            });
            app.canvas.setDirty(true, true)
        }

        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*1.3) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*1.3;
            let titleHeight = LiteGraph.NODE_TITLE_HEIGHT;
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            let buttonWidth = cWidth-titleHeight-6;
            let cx = (this.flags.collapsed?cWidth:this.size[0])-buttonWidth-6;
            ctx.fillStyle = this.color || LiteGraph.NODE_DEFAULT_COLOR;
            ctx.beginPath();
            ctx.rect(cx, 2-titleHeight, buttonWidth, titleHeight-4);
            ctx.fill();
            cx += buttonWidth/2;
            ctx.lineWidth = 1;
            if (this.mouseOver)
            {
                ctx.fillStyle = LiteGraph.NODE_SELECTED_TITLE_COLOR
                ctx.beginPath(); ctx.moveTo(cx-8,-titleHeight/2-8); ctx.lineTo(cx+0,-titleHeight/2); ctx.lineTo(cx-8,-titleHeight/2+8); ctx.fill();
                ctx.beginPath(); ctx.moveTo(cx+1,-titleHeight/2-8); ctx.lineTo(cx+9,-titleHeight/2); ctx.lineTo(cx+1,-titleHeight/2+8); ctx.fill();
            }
            else
            {
                ctx.fillStyle = (this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR);
                ctx.beginPath(); ctx.rect(cx-10,-titleHeight/2-8, 4, 16); ctx.fill();
                ctx.beginPath(); ctx.rect(cx-2,-titleHeight/2-8, 4, 16); ctx.fill();
            }
        }

        this.node.computeSize = function()
        {
            return [ 200, LiteGraph.NODE_SLOT_HEIGHT*1.3 ];
        }
    }
}

app.registerExtension(
{
    name: "Runnodeselected",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "Runnodeselected")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.runnode = new Run_node(this);
            }
        }
    }
});
