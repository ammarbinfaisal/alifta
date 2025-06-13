import { Suspense } from "react";

export function Layout({
    children
}: {
    children: React.ReactNode
}) {
    return (
        <Suspense fallback={"..."}>
            {children}
        </Suspense>
    )
}