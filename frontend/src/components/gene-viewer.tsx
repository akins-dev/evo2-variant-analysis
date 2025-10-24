import type { GeneFromSearch } from "@/utils/genome-api";
import { Button } from "./ui/button";
import { ArrowLeft } from "lucide-react";

const GeneViewer = ({
  gene,
  genomeId,
  onClose,
}: {
  gene: GeneFromSearch;
  genomeId: string;
  onClose: () => void;
}) => {
  return (
    <div className="space-y-6">
      <Button
        variant="ghost"
        className="text-primary hover:bg-secondary/70 cursor-pointer"
        size="sm"
        onClick={onClose}
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to results
      </Button>
    </div>
  );
};

export default GeneViewer;
