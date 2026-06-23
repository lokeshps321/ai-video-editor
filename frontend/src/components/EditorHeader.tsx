import BrandLogo from "./BrandLogo";
import "./EditorHeader.css";

type EditorHeaderProps = {
  title: string;
};

export function EditorHeader({ title }: EditorHeaderProps) {
  return (
    <header className="topBar">
      <div className="headerLogos">
        <BrandLogo variant="header" linkTo="/" />
        <div className="headerProjectTitle">
          <span className="headerProjectDivider" aria-hidden="true">
            /
          </span>
          <h1>{title}</h1>
        </div>
      </div>
    </header>
  );
}
