import { Suspense } from "react";

export function Layout({
    children
}: {
    children: React.ReactNode
}) {
    <Suspense>
        {children}
    </Suspense>
}