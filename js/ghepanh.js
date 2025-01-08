// ComfyUI.mxToolkit.Slider v.0.9.8 - Max Smirnov 2024
import { app } from "../../scripts/app.js";
class Position
{
    constructor(node)
    {
        this.node = node;
        this.node.properties = this.node.properties || {};
        this.node.properties.value=50;
        this.node.properties.min=0;
        this.node.properties.max=100;
        this.node.properties.step=5;
        this.node.properties.snap=true;
        this.node.intpos = { x:0.2 };
        this.node.size = [300, Math.floor(LiteGraph.NODE_SLOT_HEIGHT*7*1.5)];
        const fontsize = LiteGraph.NODE_SUBTEXT_SIZE;
        const shX = (this.node.slot_start_y || 0)+fontsize*1.5;
        const shY = LiteGraph.NODE_SLOT_HEIGHT*7;
        const shiftLeft = 55;
        const shiftRight = 55;
        this.node.widgets[2].hidden = true;
        //this.node.widgets[3].hidden = true;
        this.node.onAdded = function ()
        {
            this.widgets_start_y = 50;
            this.intpos.x = Math.max(0, Math.min(1, (this.properties.value-this.properties.min)/(this.properties.max-this.properties.min)));
            //if (this.size) if (this.size.length) if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*5*1.5) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*5*1.5;
        }
        this.node.onGraphConfigured = function ()
        {
            this.configured = true;
            this.onPropertyChanged();
        }
        this.node.onPropertyChanged = function (propName)
        {
            if (!this.configured) return;
            if (this.properties.step <= 0) this.properties.step = 1;
            if ( isNaN(this.properties.value) ) this.properties.value = this.properties.min;
            if ( this.properties.min >= this.properties.max ) this.properties.max = this.properties.min+this.properties.step;
            if ((propName === "min") && (this.properties.value < this.properties.min)) this.properties.value = this.properties.min;
            if ((propName === "max") && (this.properties.value > this.properties.max)) this.properties.value = this.properties.max;
            this.properties.value = Math.round(Math.pow(10,0)*this.properties.value)/Math.pow(10,0);
            this.intpos.x = Math.max(0, Math.min(1, (this.properties.value-this.properties.min)/(this.properties.max-this.properties.min)));
            this.widgets[2].value = Math.floor(this.properties.value);
        }
        this.node.onConnectionsChange = function (type, index, connected, link_info)
        {
            if (link_info)
            {
                if (connected)
                {
                    if (type === LiteGraph.INPUT)
                    {
                        const cnode = app.graph.getNodeById(link_info.origin_id);
                        const ctype = cnode.outputs[link_info.origin_slot].type;
                        const color = LGraphCanvas.link_type_colors[ctype];
                        this.outputs[0].type = ctype;
                        this.outputs[0].name = ctype;
                        this.inputs[0].type = ctype;
                        if (link_info.id) { app.graph.links[link_info.id].color = color; }
                        if (this.outputs[0].links !== null)
                            for (let i = this.outputs[0].links.length; i > 0; i--)
                            {
                                const tlinkId = this.outputs[0].links[i-1];
                                const tlink = app.graph.links[tlinkId];
                                if (this.configured) if ( ctype !== tlink.type ) app.graph.getNodeById(tlink.target_id).disconnectInput(tlink.target_slot);
                            }
                    }
                    if (type === LiteGraph.OUTPUT && this.inputs[0].link === null)
                    {
                        this.inputs[0].type = link_info.type;
                        this.outputs[0].type = link_info.type;
                        this.outputs[0].name = link_info.type;
                    }
                }
                else if ((( type === LiteGraph.INPUT ) && ( this.outputs[0].links === null || this.outputs[0].links.length === 0 )) || (( type === LiteGraph.OUTPUT) && ( this.inputs[0].link === null ))) this.onAdded();
            }
            this.computeSize();
        };
        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if ( this.flags.collapsed ) return false;
            //if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*5*1.5) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*5*1.5;
            let dgt = parseInt(0);
            ctx.fillStyle="rgba(20,20,20,0.5)";
            ctx.beginPath();
            ctx.roundRect( shiftLeft, shY-1, this.size[0]-shiftRight-shiftLeft, 4, 2);
            ctx.fill();
            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.beginPath();
            ctx.arc(shiftLeft+(this.size[0]-shiftRight-shiftLeft)*this.intpos.x, shY+1, 7, 0, 2 * Math.PI, false);
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle=node.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR;
            ctx.beginPath();
            ctx.arc(shiftLeft+(this.size[0]-shiftRight-shiftLeft)*this.intpos.x, shY+1, 5, 0, 2 * Math.PI, false);
            ctx.stroke();
            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.font = (fontsize) + "px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Trái", 30, shY+1);
            ctx.fill();
            ctx.fillText("Phải", this.size[0]-30, shY+1);
            ctx.fill();
            ctx.fillText(this.properties.value.toFixed(dgt), this.size[0]/2, shY+20);
        }
        this.node.onDblClick = function(e, pos, canvas)
        {
            if ( e.canvasX > this.pos[0]+this.size[0]-shiftRight+10 )
            {
                canvas.prompt("value", this.properties.value, function(v) {if (!isNaN(Number(v))) { this.properties.value = Number(v); this.onPropertyChanged("value");}}.bind(this), e);
                return true;
            }
        }
        this.node.onMouseDown = function(e)
        {
            if ( e.canvasY - this.pos[1] < 0 ) return false;
            if ( e.canvasX < this.pos[0]+shiftLeft-5 || e.canvasX > this.pos[0]+this.size[0]-shiftRight+5 ) return false;
            if ( e.canvasY < this.pos[1]+20-5 || e.canvasY > this.pos[1]+this.size[1]-20+5) return false;
            this.capture = true;
            this.unlock = false;
            this.captureInput(true);
            this.valueUpdate(e);
            return true;
        }
        this.node.onMouseMove = function(e)
        {
            if (!this.capture) return;
            this.valueUpdate(e);
        }
        this.node.onMouseUp = async function(e)
        {
            if (!this.capture) return;
            this.capture = false;
            this.captureInput(false);
            this.widgets[2].value = Math.floor(this.properties.value);
            const graph = window.graph;
            const nodes = graph._nodes;
            const targetNode = nodes.find(node => node.id === this.id);
            nodes.forEach(node => {
                node.mode = LiteGraph.NEVER;
            });
            nodes.forEach(node => {
                if (node.id == this.id) {
                    node.inputs.forEach(input => {
                        if (input.link) {
                            const tlink = app.graph.links[input.link]
                            const linkedNode = graph.getNodeById(tlink.origin_id);
                            linkedNode.mode = LiteGraph.ALWAYS;
                        }
                    });
                    node.mode = LiteGraph.ALWAYS;
                }
            });
            await app.queuePrompt(0);
            nodes.forEach(node => {
                node.mode = LiteGraph.ALWAYS;
            });
        }
        this.node.valueUpdate = function(e)
        {
            let prevX = this.properties.value;
            let rn = Math.pow(10,0);
            let vX = (e.canvasX - this.pos[0] - shiftLeft)/(this.size[0]-shiftRight-shiftLeft);
            if (e.ctrlKey) this.unlock = true;
            if (e.shiftKey !== this.properties.snap)
            {
                let step = this.properties.step/(this.properties.max - this.properties.min);
                vX = Math.round(vX/step)*step;
            }
            this.intpos.x = Math.max(0, Math.min(1, vX));
            this.properties.value = Math.round(rn*(this.properties.min + (this.properties.max - this.properties.min) * ((this.unlock)?vX:this.intpos.x)))/rn;
            this.updateThisNodeGraph?.();
            if ( this.properties.value !== prevX ) this.graph.setisChangedFlag(this.id);
        }
        this.node.onSelected = function(e) { this.onMouseUp(e) }
        //this.node.computeSize = () => [LiteGraph.NODE_WIDTH,Math.floor(LiteGraph.NODE_SLOT_HEIGHT*5*1.5)];
    }
}

app.registerExtension(
{
    name: "ghepanh",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "ghepanh")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.ghepanh = new Position(this);
            }
        }
    }
});
