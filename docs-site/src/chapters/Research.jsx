import { researchCards } from "../siteData.js";
import { ChapterSection } from "../ui.jsx";

export default function ResearchChapter() {
  return (
    <>
      <ChapterSection
        eyebrow="Chapter 7"
        title="Research references that shape the MELD story"
        body="These cards explain not only which papers matter, but how the framework uses them."
      >
        <div className="research-grid">
          {researchCards.map((card) => (
            <article className="research-card" key={card.title}>
              <h3>{card.title}</h3>
              <p className="research-citation">{card.citation}</p>
              <div className="research-usage">
                <span>How MELD uses it</span>
                <p>{card.usage}</p>
              </div>
            </article>
          ))}
        </div>
      </ChapterSection>
    </>
  );
}
