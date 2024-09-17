/* globals echarts, UMAP, tsnejs */
import { pipeline } from "./transformers.js";

// https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main/onnx

const pipePromise = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
async function getEmbeddingFromText(text) {
  const pipe = await pipePromise;
  const output = await pipe(text, {
    pooling: "mean",
    normalize: true,
  });
  return Array.from(output.data);
}

const LINES = [
  "Moving Beyond Syntax: Lessons from 20 Years of Blocks Programming in AgentSheets by Alexander Repenning",
  "Pygmalion by David C. Smith",
  "Elephant in the Room",
  "Beyond Efficiency by Dave Ackley",
  "Myths & Mythconceptions by Mary Shaw",
  "Propositions as Types by Philip Wadler",
  "Considered Harmful",
  "A Small Matter of Programming by Bonnie Nardi",
  "Interpreting the Rule(s) of Code by Laurence Diver",
  "INTERCAL by Donald Woods & James Lyon",
  "Out of the Tar Pit by Ben Moseley & Peter Marks",
  "No Silver Bullet by Fred Brooks",
  "Programming as Theory Building by Peter Naur",
  "Magic Ink by Bret Victor",
  "Worse is Better by Richard P. Gabriel",
  "Structure of a Programming Language Revolution by Richard P. Gabriel",
  "Personal Dynamic Media by Alan Kay & Adele Goldberg",
  "Augmenting Human Intellect by Doug Engelbart",
  "Man-Computer Symbiosis by J.C.R. Licklider",
  "Ella Hoeppner: Vlojure",
  "Scott Anderson: End-user Programming in VR",
  "Amjad Masad: Replit",
  "Toby Schachman: Cuttle, Apparatus, and Recursive Drawing",
  "Mary Rose Cook: Isla & Code Lauren",
  "Ravi Chugh: Sketch-n-Sketch",
  "Jennifer Jacobs: Para & Dynamic Brushes",
  "Max/MSP & Pure Data: Miller Puckette",
  "2020 Community Survey",
  "Orca: Devine Lu Linvega",
  "Making Your Own Tools: Devine Lu Linvega",
  "Unveiling Dark: Ellen Chisa & Paul Biggar",
  "Blurring the Line Between User and Programmer: Lane Shackleton",
  "The Aesthetics of Programming Tools: Jack Rusher",
  "Joining Logic, Relational, and Functional Programming: Michael Arntzenius",
  "Mathematical Foundations for the Activity of Programming: Cyrus Omar",
  "The Case for Formal Methods: Hillel Wayne",
  "De-Nerding Programming: Jonathan Edwards",
  "Moldable Development: Tudor Girba",
  "Democratizing Web Design: Vlad Magdalin",
  "The Edges of Representation: Katherine Ye",
  "Reflection 14: /about",
  "Basic Developer Human Rights: Quinn Slack",
  "Sustaining the Underfunded: Nadia Eghbal",
  "On The Maintenance Of Large Software: James Koppel",
  "Reflection Thirteen - Independent mentorship",
  "Exploring Dynamicland - Omar Rizwan",
  "Bringing Explicit Modeling To The Web: David K Piano",
  "Compassion & Programming: Glen Chiacchieri",
  "You Should Consider Some States Kevin Lynagh",
  "Stop Being A Sysadmin For Your Own Machine: Nick Santos",
  "Teaching Abstraction: Brent Yorgey",
  "Learning Programming At Scale: Philip Guo",
  "Building for Developers: Aidan Cunniffe",
  "Coding On (the) Beach: Jason Brennan",
  "Building Universe: Joe Cohen",
  "Research Recap Nine: Constructing My Crusade",
  "Bootstrapping Bubble.is: Emmanuel Straschnov",
  "Research Recap Eight: Life & Work Planning",
  "Raising Genius with Scott Mueller",
  "Research Recap Seven - Master Planning",
  "Teaching Elm To 4th Graders: Christopher Anand",
  "Research Recap Six: CycleJS Deep Dive",
  "How ReactJS was created - with Pete Hunt",
  "Unison's Paul Chiusano on how Abstraction Will Save Distributed Computing",
  "Research Recap Five",
  "Research Recap Four",
  "Looker's Lloyd Tabb on Growing Languages Through Deprecation",
  "Research Recap Three (WoofJS Workflow)",
  "Samantha John Of Hopscotch On Learnable Programming",
  "Research Recap Two",
  "Jonathan Leung on Inventing on Principle",
  "Research Recap - A Year in Review",
  "Welcome to the Future of Coding",
];

export async function main() {
  console.log("init");
  setupList();
  drawTextLinesUMAP(LINES);
}

async function drawTextLinesTSNE(lines) {
  draw(await linesTo3d(lines), lines);
}

async function drawTextLinesUMAP(lines, opts = {}) {
  const data = [];

  for (const line of lines) {
    data.push(await getEmbeddingFromText(line));
  }
  const epochs = opts.epochs ?? 500;
  // https://github.com/PAIR-code/umap-js?tab=readme-ov-file#parameters
  const umap = new UMAP.UMAP({
    nComponents: 3,
    nEpochs: epochs,
    nNeighbors: opts.perplexity ?? 15,
  });
  const embedding = await umap.fit(data);
  draw(embedding, lines);
}

async function linesTo3d(lines, opts = {}) {
  const opt = {
      epsilon: 10, // epsilon is learning rate (10 = default)
      perplexity: opts.perplexity ?? 15, // roughly how many neighbors each point influences (30 = default)
      dim: 3, // dimensionality of the embedding (2 = default)
    },
    tsne = new tsnejs.tSNE(opt);
  const dists = [];

  for (const line of lines) {
    dists.push(await getEmbeddingFromText(line));
  }

  tsne.initDataRaw(dists);

  const epochs = opts.epochs ?? 500;
  for (let k = 0; k < epochs; k++) {
    tsne.step(); // every time you call this, solution gets better
  }

  return tsne.getSolution();
}

let chart;
function draw(data, labels) {
  if (!chart) {
    const chartDom = document.getElementById("main");
    chart = echarts.init(chartDom);
  }

  const option = {
    grid3D: {},
    xAxis3D: {
      type: "value",
      min: "dataMin",
      max: "dataMax",
    },
    yAxis3D: {
      type: "value",
      min: "dataMin",
      max: "dataMax",
    },
    zAxis3D: {
      type: "value",
      min: "dataMin",
      max: "dataMax",
    },
    tooltip: {
      formatter: function (params) {
        //console.log(params);
        return labels[params.dataIndex];
      },
    },
    series: [
      {
        type: "scatter3D",
        //symbolSize: 50,
        data,
        //itemStyle: { opacity: 1, },
      },
    ],
  };

  chart.setOption(option);
}

function setupList() {
  const itemsContainer = document.getElementById("items");
  const addButton = document.querySelector(".add-btn");
  const clearButton = document.querySelector(".clear-btn");
  const calculateButton = document.querySelector(".calculate-btn");
  const epochsInput = document.querySelector("#epochs");
  const perplexityInput = document.querySelector("#perplexity");
  const algorithmSelect = document.querySelector("#algorithm");

  function addInput(text) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("input-wrapper");

    const input = document.createElement("input");
    input.type = "text";
    input.value = text;

    const removeBtn = document.createElement("button");
    removeBtn.classList.add("remove-btn");
    removeBtn.textContent = "X";

    wrapper.appendChild(input);
    wrapper.appendChild(removeBtn);

    itemsContainer.appendChild(wrapper);

    removeBtn.addEventListener("click", () => {
      wrapper.remove();
    });
  }

  function getOpts() {
    return {
      epochs: parseIntOr(epochsInput.value, 500),
      perplexity: parseIntOr(perplexityInput.value, 15),
      algorithm: algorithmSelect.value,
    };
  }

  function clearList() {
    itemsContainer.innerHTML = "";
  }

  function calculate() {
    const opts = getOpts();
    console.log("calculate!", opts);
    const lines = [...itemsContainer.querySelectorAll("input")].map(
      (n) => n.value,
    );
    if (opts.algorithm === "t-SNE") {
      drawTextLinesTSNE(lines);
    } else {
      drawTextLinesUMAP(lines);
    }
  }

  addButton.addEventListener("click", (_) => addInput(""));
  clearButton.addEventListener("click", (_) => clearList());
  calculateButton.addEventListener("click", (_) => calculate());

  for (const line of LINES) {
    addInput(line);
  }
}

function parseIntOr(v, d) {
  const r = parseInt(v, 10);
  return Number.isFinite(r) ? r : d;
}

main();
