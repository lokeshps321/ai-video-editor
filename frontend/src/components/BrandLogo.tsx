import { Link } from "react-router-dom";
import { Zap } from "lucide-react";
import { BRAND } from "../config/brand";

type BrandLogoProps = {
    variant?: "nav" | "footer" | "header";
    linkTo?: string | null;
    className?: string;
};

const iconSizes = {
    nav: 25,
    footer: 22,
    header: 25,
} as const;

export default function BrandLogo({
    variant = "nav",
    linkTo = "/",
    className = "",
}: BrandLogoProps) {
    const classes = ["brand-logo", `brand-logo-${variant}`, className].filter(Boolean).join(" ");
    const content = (
        <>
            <Zap className="brand-logo-icon" size={iconSizes[variant]} aria-hidden="true" />
            <span className="brand-logo-text">{BRAND.productName}</span>
        </>
    );

    if (linkTo === null) {
        return <div className={classes}>{content}</div>;
    }

    return (
        <Link to={linkTo} className={classes} aria-label={`${BRAND.productName} home`}>
            {content}
        </Link>
    );
}
