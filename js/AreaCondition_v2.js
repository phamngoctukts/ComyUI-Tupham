import { app } from "/scripts/app.js";
import {CUSTOM_INT, transformFunc,getDrawColor, computeCanvasSize} from "./utils.js"

function addMultiAreaConditioningCanvas(node, app) {
	let dk = 0;
	node.setDirtyCanvas(true);
	const widget = {
		type: "customCanvas",
		name: "AreaCondition_v2-Canvas",
		get value() {
			return this.canvas.value;
		},
		set value(x) {
			this.canvas.value = x;
		},

		draw: function (ctx, node, widgetWidth, widgetY) {			
			// If we are initially offscreen when created we wont have received a resize event
			// Calculate it here instead

			if (!node.canvasHeight) {
				computeCanvasSize(node, node.size)
			}
			const visible = true //app.canvasblank.ds.scale > 0.5 && this.type === "customCanvas";
			const t = ctx.getTransform();
			const margin = 10
			const border = 2
			const widgetHeight = node.canvasHeight
            let values = node.properties["values"]
			const index = Math.round(node.widgets[node.index].value)
			let vt = node.widgets[3].value
			if (node.widgets[3].value > values.length-1) {
				node.widgets[3].value = values.length-1; 
				vt = values.length-1;
				return;
			}//node.widgets[3].value = v
			node.properties["width"] = node.widgets[0].value;
			node.properties["height"] = node.widgets[1].value;
			node.widgets[2].computedHeight = 100;
			if (dk === vt){
				node.properties["values"][vt][0] = node.widgets[2].value;
				node.properties["values"][vt][1]= node.widgets[4].value;
				node.properties["values"][vt][2]= node.widgets[5].value;
				node.properties["values"][vt][3]= node.widgets[6].value;
				node.properties["values"][vt][4] = node.widgets[7].value;
				node.properties["values"][vt][5] = node.widgets[8].value;
			} else {
				values = node.properties["values"];
				node.widgets[2].value = values[vt][0];
				node.widgets[4].value = values[vt][1];
				node.widgets[5].value = values[vt][2];
				node.widgets[6].value = values[vt][3];
				node.widgets[7].value = values[vt][4];
				node.widgets[8].value = values[vt][5];
			}
			const width = Math.round(node.properties["width"])
			const height = Math.round(node.properties["height"])
			const scale = Math.min((widgetWidth-margin*2)/width, (widgetHeight-margin*2)/height)
			Object.assign(this.canvas.style, {
				left: `${t.e}px`,
				top: `${t.f + (widgetY*t.d)}px`,
				width: `${widgetWidth * t.a}px`,
				height: `${widgetHeight * t.d}px`,
				position: "absolute",
				zIndex: 1,
				fontSize: `${t.d * 10.0}px`,
				pointerEvents: "none",
			});
			this.canvas.hidden = !visible;
            let backgroudWidth = width * scale
            let backgroundHeight = height * scale
			let xOffset = margin
			if (backgroudWidth < widgetWidth) {
				xOffset += (widgetWidth-backgroudWidth)/2 - margin
			}
			let yOffset = margin
			if (backgroundHeight < widgetHeight) {
				yOffset += (widgetHeight-backgroundHeight)/2 - margin
			}
			let widgetX = xOffset
			widgetY = widgetY + yOffset
			ctx.fillStyle = "#000000"
			ctx.fillRect(widgetX-border, widgetY-border, backgroudWidth+border*2, backgroundHeight+border*2)
			ctx.fillStyle = globalThis.LiteGraph.NODE_DEFAULT_BGCOLOR
			ctx.fillRect(widgetX, widgetY, backgroudWidth, backgroundHeight);
			function getDrawArea(v) {
				let x = v[1]*backgroudWidth/width
				let y = v[2]*backgroundHeight/height
				let w = v[3]*backgroudWidth/width
				let h = v[4]*backgroundHeight/height
				if (x > backgroudWidth) { x = backgroudWidth}
				if (y > backgroundHeight) { y = backgroundHeight}
				if (x+w > backgroudWidth) {
					w = Math.max(0, backgroudWidth-x)
				}				
				if (y+h > backgroundHeight) {
					h = Math.max(0, backgroundHeight-y)
				}
				return [x, y, w, h]
			}           
			// Draw all the conditioning zones
			for (const [k, v] of values.entries()) {
				if (k == index) {continue}
				const [x, y, w, h] = getDrawArea(v)
				ctx.fillStyle = getDrawColor(k/values.length, "80") //colors[k] + "B0"
				ctx.fillRect(widgetX+x, widgetY+y, w, h)
			}
			ctx.beginPath();
			ctx.lineWidth = 1;
			for (let x = 0; x <= width/64; x += 1) {
				ctx.moveTo(widgetX+x*64*scale, widgetY);
				ctx.lineTo(widgetX+x*64*scale, widgetY+backgroundHeight);
			}
			for (let y = 0; y <= height/64; y += 1) {
				ctx.moveTo(widgetX, widgetY+y*64*scale);
				ctx.lineTo(widgetX+backgroudWidth, widgetY+y*64*scale);
			}
			ctx.strokeStyle = "#00000050";
			ctx.stroke();
			ctx.closePath();
			// Draw currently selected zone
			console.log(index)
			let [x, y, w, h] = getDrawArea(values[index])
			w = Math.max(32*scale, w)
			h = Math.max(32*scale, h)
			ctx.fillStyle = "#ffffff"
			ctx.fillRect(widgetX+x, widgetY+y, w, h)
			const selectedColor = getDrawColor(index/values.length, "FF")
			ctx.fillStyle = selectedColor
			ctx.fillRect(widgetX+x+border, widgetY+y+border, w-border*2, h-border*2)
			// Display
			ctx.beginPath();
			ctx.lineWidth = 1;
			ctx.strokeStyle = "white";
			ctx.stroke();
			ctx.lineWidth = 1;
			ctx.closePath();
			dk = vt;
		},
	};
	widget.canvas = document.createElement("canvas");
	widget.canvas.className = "tupham-custom-canvas";
	widget.parent = node;
	document.body.appendChild(widget.canvas);
	node.addCustomWidget(widget);
	app.canvas.onDrawBackground = function () {
		for (let n in app.graph._nodes) {
			n = app.graph._nodes[n];
			for (let w in n.widgets) {
				let wid = n.widgets[w];
				if (Object.hasOwn(wid, "canvas")) {
					wid.canvas.style.left = -8000 + "px";
					wid.canvas.style.position = "absolute";
				}
			}
		}
	};
	node.onResize = function (size) {
		computeCanvasSize(node, size);
	}

	return { minWidth: 200, minHeight: 250, widget }
}

app.registerExtension({
	name: "AreaCondition_v2",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "AreaCondition_v2") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.setProperty("width", 512)
				this.setProperty("height", 512)
				this.setProperty("values", [["", 0, 0, 0, 0, 1.0]])
				this.widgets[2].value = "";
				this.selected = false
                this.serialize_widgets = true;
				this.index = 3
				
				addMultiAreaConditioningCanvas(this, app)
				this.getExtraMenuOptions = function(_, options) {
					options.unshift(
						{
							content: `insert prompt above ${this.widgets[3].value} /\\`,
							callback: () => {
								const index = this.widgets[3].value;
								if (index >= 0) { // Nếu không phải đầu vào đầu tiên
									this.properties["values"].splice(index, 0,["", 0, 0, 0, 0, 1.0])
									this.setDirtyCanvas(true);
								}
							},
						},
						{
							content: `insert prompt below ${this.widgets[3].value} \\/`,
							callback: () => {
								const index = this.widgets[3].value;
								if (index <= this.properties["values"].length - 1) { // Nếu không phải đầu vào cuối
									this.properties["values"].splice(index+1, 0, ["", 0, 0, 0, 0, 1.0])
									this.setDirtyCanvas(true);
								}
							},
						},
						{
							content: `swap with prompt above ${this.widgets[3].value} /\\`,
							callback: () => {
								const index = this.widgets[3].value
								this.properties["values"].splice(index-1,0,this.properties["values"].splice(index,1)[0]);
								this.setDirtyCanvas(true);
							},
						},
						{
							content: `swap with prompt below ${this.widgets[3].value} \\/`,
							callback: () => {
								const index = this.widgets[3].value
								this.properties["values"].splice(index+1,0,this.properties["values"].splice(index,1)[0]);
								this.setDirtyCanvas(true);
							},
						},
						{
							content: `remove currently selected prompt ${this.widgets[3].value}`,
							callback: () => {
								const index = this.widgets[3].value;
								this.properties["values"].splice(index,1);
								this.setDirtyCanvas(true);
							},
						},
						{
							content: "remove all prompt",
							callback: () => {
								for (let i = 0; i < this.properties["values"].length; i++) {
									this.properties["values"].splice(i,1);
								}
								this.properties["values"].splice(0, 0, ["", 0, 0, 0, 0, 1.0])
								this.setDirtyCanvas(true);
							},
						},
					);
				}
				this.onDrawBackground = function(ctx)
				{
					this.configured = true;
					ctx.fillStyle="rgba(20,20,20,0.5)";
					ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
					ctx.font = "25px Arial";
					ctx.textAlign = "center";
					ctx.fillText(this.properties["values"].length, 50, 50);
					//app.setDirtyCanvas(true, true);
				}
				this.onRemoved = function () {
					// When removing this node we need to remove the input from the DOM
					for (let y in this.widgets) {
						if (this.widgets[y].canvas) {
							this.widgets[y].canvas.remove();
						}
					}
					//app.setDirtyCanvas(true, true);
				};			
				this.onSelected = function () {
					this.selected = true
				}
				this.onDeselected = function () {
					this.selected = false
				}
				return r;
			};
		}
	},
});