import { useState } from "react";
import { installCommands } from "../siteData.js";
import { ChapterSection, CodeBlock, StatCard } from "../ui.jsx";

export default function InstallationChapter() {
  const [tab, setTab] = useState("powershell");

  return (
    <>
      <ChapterSection
        eyebrow="Fast package install"
        title="Use pip when you just want MELD without editing the repo"
        body="If you only want to use the framework and not work on the source tree, install the package first and then use the CLI or Python API directly."
      >
        <CodeBlock
          label="Package install"
          code={"python -m pip install --upgrade pip\npython -m pip install meld-framework"}
        />
      </ChapterSection>

      <ChapterSection
        eyebrow="Chapter 1"
        title="Install MELD and make the environment complete"
        body="Activation alone does not install packages. The commands below are the full setup path, including bootstrap downloads."
      >
        <div className="tab-strip">
          <button type="button" className={tab === "powershell" ? "is-active" : ""} onClick={() => setTab("powershell")}>
            Windows PowerShell
          </button>
          <button type="button" className={tab === "cmd" ? "is-active" : ""} onClick={() => setTab("cmd")}>
            Windows CMD
          </button>
          <button type="button" className={tab === "unix" ? "is-active" : ""} onClick={() => setTab("unix")}>
            macOS / Linux
          </button>
        </div>
        <CodeBlock
          label={tab === "unix" ? "Shell setup" : tab === "cmd" ? "CMD setup" : "PowerShell setup"}
          code={installCommands[tab]}
        />
      </ChapterSection>

      <ChapterSection
        eyebrow="What pip install does"
        title="The source install pulls the runtime dependencies too"
        body="python -m pip install . installs MELD and the dependencies declared by the project, including torch, torchvision, continuum, transformers, datasets, fastapi, and uvicorn."
      >
        <div className="bullet-grid">
          <StatCard title="Virtualenv first" body="Create .venv before installing so the package, datasets, and CLI tools are isolated inside the repo workflow." />
          <StatCard title="Bootstrap next" body="Use python -m meld.bootstrap to download CIFAR assets before the first real benchmark run." />
          <StatCard title="Package install" body="Use pip install meld-framework when you only want MELD as a package, and pip install . when you are working from the source repo." />
          <StatCard title="Dataset step" body="Bootstrapping benchmark data is separate from package installation, so do it once before the first real CIFAR run." />
        </div>
      </ChapterSection>
    </>
  );
}
