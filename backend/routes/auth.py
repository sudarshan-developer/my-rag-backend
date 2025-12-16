from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import os

from database import get_db
from models import User
from .auth import get_current_user  # Your JWT helper

router = APIRouter()

@router.put("/profile")
async def update_profile(
    display_name: str = Form(...),
    bio: str = Form(default=""),
    picture: UploadFile = File(None),
    user_email: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),  # ← ADDED: Required for database
):
    """Update user profile with optional picture upload."""
    try:
        # ✅ Get user from database using email
        stmt = select(User).where(User.email == user_email)
        result = await db.execute(stmt)
        current_user = result.scalar_one_or_none()

        if not current_user:
            raise HTTPException(status_code=404, detail="User not found")

        # ✅ Validate display_name is not empty
        if not display_name or not display_name.strip():
            raise HTTPException(
                status_code=400, 
                detail="Display name cannot be empty"
            )

        # ✅ Create uploads folder if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # ✅ Handle picture upload with full validation
        picture_url = current_user.picture
        if picture:
            # Validate file type - images only
            if not picture.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail="File must be an image (JPG, PNG, GIF, etc.)"
                )

            # Read file and validate size (5MB max)
            contents = await picture.read()
            if len(contents) > 5 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail="Image too large (maximum 5MB)"
                )

            # Delete old picture if exists
            if current_user.picture and current_user.picture.startswith("/uploads/"):
                old_path = current_user.picture.lstrip("/")
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"🗑️ Deleted old image: {old_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not delete old image: {e}")

            # Save new picture with unique filename
            file_extension = picture.filename.split('.')[-1].lower()
            filename = f"profile_{current_user.id}.{file_extension}"
            filepath = f"uploads/{filename}"

            with open(filepath, "wb") as f:
                f.write(contents)

            picture_url = f"/uploads/{filename}"
            print(f"✅ Saved new image: {picture_url}")

        # ✅ Update user object
        current_user.display_name = display_name.strip()
        current_user.bio = bio or ""
        current_user.picture = picture_url
        current_user.updated_at = datetime.utcnow()  # Track when updated

        # ✅ Use async database operations
        db.add(current_user)
        await db.commit()
        await db.refresh(current_user)

        # ✅ Return complete response
        return {
            "id": current_user.id,
            "email": current_user.email,
            "display_name": current_user.display_name,
            "bio": current_user.bio,
            "picture": current_user.picture,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "updated_at": current_user.updated_at.isoformat() if current_user.updated_at else None,
            "message": "Profile updated successfully"
        }

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Catch any other errors
        await db.rollback()
        print(f"❌ Error updating profile: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update profile: {str(e)}"
        )
