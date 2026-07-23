from app.services.nvidia_service import NvidiaService


def main() -> None:
    service = NvidiaService()

    elements = [
        {
            "id": "element_1",
            "type": "rounded_rectangle",
            "text": "Start",
            "bbox": [100, 50, 250, 120],
        },
        {
            "id": "element_2",
            "type": "rectangle",
            "text": "Upload Image",
            "bbox": [100, 170, 250, 240],
        },
        {
            "id": "element_3",
            "type": "diamond",
            "text": "Image Valid?",
            "bbox": [100, 290, 250, 380],
        },
        {
            "id": "element_4",
            "type": "rounded_rectangle",
            "text": "Generate Diagram",
            "bbox": [100, 430, 250, 500],
        },
    ]

    connections = [
        {
            "source": "element_1",
            "target": "element_2",
        },
        {
            "source": "element_2",
            "target": "element_3",
        },
        {
            "source": "element_3",
            "target": "element_4",
            "label": "Yes",
        },
    ]

    mermaid = service.generate_mermaid(
        elements=elements,
        connections=connections,
    )

    print("\nGenerated Mermaid:\n")
    print(mermaid)


if __name__ == "__main__":
    main()